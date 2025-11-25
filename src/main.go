// main_optimized.go - Optimized batch font renderer
// Improvements:
// - Precise glyph bounding via font.BoundString to minimize glyph size
// - Simple skyline (shelf) packing with glyphs sorted by height desc
// - Pre-baked glyph quad metrics (no per-frame bounds calc)
// - Vertex format: position(2) + uv(2) (4 floats). Color moved to uniform and batching by color.
// - Single VBO allocated once; use BufferSubData to update the used portion each frame
// - Reduced allocations and string formatting frequency

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"

	"go-font-render/packages/gl/v3.3-core/gl"
	"go-font-render/packages/glfw"

	"github.com/golang/freetype/truetype"
	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// ---- config ----
var (
	windowWidth  = 800
	windowHeight = 600

	atlasWidth  = 512
	atlasHeight = 256i

	fontPath    string
	fontDPI     = 72
	fontPixelEm = 32
	padding     = 1

	useVSync = true
)

// Simple FPS counter
type FPSCounter struct {
	frames   int
	lastTime time.Time
	fps      float64
}

func NewFPSCounter() *FPSCounter {
	return &FPSCounter{lastTime: time.Now()}
}

func (f *FPSCounter) Update() {
	f.frames++
	if time.Since(f.lastTime) >= time.Second {
		f.fps = float64(f.frames) / time.Since(f.lastTime).Seconds()
		f.frames = 0
		f.lastTime = time.Now()
	}
}

func (f *FPSCounter) GetFPS() float64 { return f.fps }

// Glyph holds precomputed metadata and the rasterized alpha bytes.
type Glyph struct {
	Rune     rune
	W, H     int
	AtlasX   int
	AtlasY   int
	Advance  int // in pixels
	BearingX int
	BearingY int
	U1, V1   float32
	U2, V2   float32
	Pixels   []uint8 // W*H alpha
}

// Skyline shelf packer (simple and effective for fonts)
type ShelfPacker struct {
	W, H int
	x    int
	y    int
	rowH int
}

func NewShelfPacker(w, h int) *ShelfPacker { return &ShelfPacker{W: w, H: h} }

func (s *ShelfPacker) Pack(wid, hei int) (int, int, bool) {
	if wid > s.W || hei > s.H {
		return -1, -1, false
	}
	if s.x+wid > s.W {
		// next row
		s.x = 0
		s.y += s.rowH
		s.rowH = 0
	}
	if s.y+hei > s.H {
		return -1, -1, false
	}
	if hei > s.rowH {
		s.rowH = hei
	}
	x, y := s.x, s.y
	s.x += wid
	return x, y, true
}

// GL atlas stores CPU copy for debug/PNG and GL texture handle
type GLAtlas struct {
	Tex uint32
	W   int
	H   int
	CPU []uint8
}

func NewGLAtlas(w, h int) *GLAtlas {
	var tex uint32
	gl.GenTextures(1, &tex)
	gl.BindTexture(gl.TEXTURE_2D, tex)
	// allocate single-channel texture
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.R8, int32(w), int32(h), 0, gl.RED, gl.UNSIGNED_BYTE, nil)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	return &GLAtlas{Tex: tex, W: w, H: h, CPU: make([]uint8, w*h)}
}

func (a *GLAtlas) UploadSubImage(x, y, w, h int, pixels []uint8) {
	if len(pixels) < w*h {
		log.Println("UploadSubImage: pixels too small")
		return
	}
	for row := 0; row < h; row++ {
		dst := (y+row)*a.W + x
		src := row * w
		copy(a.CPU[dst:dst+w], pixels[src:src+w])
	}
	gl.BindTexture(gl.TEXTURE_2D, a.Tex)
	gl.PixelStorei(gl.UNPACK_ALIGNMENT, 1)
	gl.TexSubImage2D(gl.TEXTURE_2D, 0, int32(x), int32(y), int32(w), int32(h), gl.RED, gl.UNSIGNED_BYTE, gl.Ptr(&pixels[0]))
}

func (a *GLAtlas) DumpPNG(fname string) error {
	img := image.NewRGBA(image.Rect(0, 0, a.W, a.H))
	for yy := 0; yy < a.H; yy++ {
		for xx := 0; xx < a.W; xx++ {
			v := a.CPU[yy*a.W+xx]
			img.SetRGBA(xx, yy, color.RGBA{v, v, v, 255})
		}
	}
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// Vertex size: position(2) + uv(2) = 4 floats

// TextCommand: stores text and transform + color
type TextCommand struct {
	Text string
	X, Y int
	C    [3]float32
}

// Batch renderer groups commands by color and issues one draw per color group
type BatchTextRenderer struct {
	prog     uint32
	vao, vbo uint32
	glyphs   map[rune]*Glyph
	atlas    *GLAtlas
	commands []TextCommand
	vertices []float32 // dynamic vertex buffer
	maxVerts int
	projLoc  int32
	colorLoc int32
}

func NewBatchTextRenderer(prog, vao, vbo uint32, glyphs map[rune]*Glyph, atlas *GLAtlas, projLoc, colorLoc int32) *BatchTextRenderer {
	b := &BatchTextRenderer{
		prog:     prog,
		vao:      vao,
		vbo:      vbo,
		glyphs:   glyphs,
		atlas:    atlas,
		commands: make([]TextCommand, 0, 64),
		maxVerts: 65536,
	}
	b.vertices = make([]float32, 0, b.maxVerts*4)
	b.projLoc = projLoc
	b.colorLoc = colorLoc
	return b
}

func (b *BatchTextRenderer) Draw(text string, x, y int, c [3]float32) {
	b.commands = append(b.commands, TextCommand{Text: text, X: x, Y: y, C: c})
}

// group commands by color using string key
func colorKey(c [3]float32) string {
	return fmt.Sprintf("%.5f:%.5f:%.5f", c[0], c[1], c[2])
}

func (b *BatchTextRenderer) Flush() {
	if len(b.commands) == 0 {
		return
	}

	// Group commands by color to minimize draw calls while avoiding per-vertex color
	groups := make(map[string][]TextCommand)
	keys := make([]string, 0, len(b.commands))
	for _, cmd := range b.commands {
		k := colorKey(cmd.C)
		if _, ok := groups[k]; !ok {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], cmd)
	}

	gl.BindVertexArray(b.vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, b.vbo)
	gl.ActiveTexture(gl.TEXTURE0)
	gl.BindTexture(gl.TEXTURE_2D, b.atlas.Tex)

	for _, k := range keys {
		cmds := groups[k]
		// parse color key back
		parts := strings.Split(k, ":")
		var col [3]float32
		if len(parts) == 3 {
			fmt.Sscanf(k, "%f:%f:%f", &col[0], &col[1], &col[2])
		}
		// build vertex buffer for this color group
		b.vertices = b.vertices[:0]
		for _, cmd := range cmds {
			x := float32(cmd.X)
			y := float32(cmd.Y)
			for _, ch := range cmd.Text {
				g, ok := b.glyphs[ch]
				if !ok {
					if sp, sok := b.glyphs[' ']; sok {
						x += float32(sp.Advance)
					} else {
						x += 8
					}
					continue
				}
				x0 := x + float32(g.BearingX)
				y0 := y - float32(g.BearingY)
				x1 := x0 + float32(g.W)
				y1 := y0 + float32(g.H)
				u1 := g.U1
				v1 := g.V1
				u2 := g.U2
				v2 := g.V2
				// two triangles (6 vertices) -> each vertex is pos.x,pos.y, u,v
				b.vertices = append(b.vertices,
					x0, y0, u1, v1,
					x1, y0, u2, v1,
					x1, y1, u2, v2,
					x0, y0, u1, v1,
					x1, y1, u2, v2,
					x0, y1, u1, v2,
				)
				x += float32(g.Advance)
			}
		}

		if len(b.vertices) == 0 {
			continue
		}
		// upload only the used portion
		sz := int32(len(b.vertices) * 4)
		gl.BufferSubData(gl.ARRAY_BUFFER, 0, int(sz), gl.Ptr(b.vertices))
		// set color uniform
		gl.Uniform3f(b.colorLoc, col[0], col[1], col[2])
		cnt := int32(len(b.vertices) / 4)
		gl.DrawArrays(gl.TRIANGLES, 0, cnt)
	}

	gl.BindVertexArray(0)
	// clear commands
	b.commands = b.commands[:0]
}

const vertexShaderSrc = `#version 330 core
layout(location=0) in vec2 inPos;
layout(location=1) in vec2 inUV;
out vec2 vUV;
uniform mat4 uProj;
void main(){ vUV = inUV; gl_Position = uProj * vec4(inPos, 0.0, 1.0); }
` + "\x00"

const fragmentShaderSrc = `#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
uniform vec3 uColor;
out vec4 FragColor;
void main(){ float a = texture(uTex, vUV).r; FragColor = vec4(uColor, a); }
` + "\x00"

// helper GL funcs (compile/link omitted for brevity)
func init() { runtime.LockOSThread() }

func compileShader(src string, t uint32) uint32 {
	sh := gl.CreateShader(t)
	cs, free := gl.Strs(src)
	gl.ShaderSource(sh, 1, cs, nil)
	free()
	gl.CompileShader(sh)
	var status int32
	gl.GetShaderiv(sh, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var l int32
		gl.GetShaderiv(sh, gl.INFO_LOG_LENGTH, &l)
		logstr := strings.Repeat("\x00", int(l+1))
		gl.GetShaderInfoLog(sh, l, nil, gl.Str(logstr))
		log.Fatalln("shader compile:", logstr)
	}
	return sh
}

func linkProgram(vs, fs uint32) uint32 {
	p := gl.CreateProgram()
	gl.AttachShader(p, vs)
	gl.AttachShader(p, fs)
	gl.LinkProgram(p)
	var status int32
	gl.GetProgramiv(p, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var l int32
		gl.GetProgramiv(p, gl.INFO_LOG_LENGTH, &l)
		logstr := strings.Repeat("\x00", int(l+1))
		gl.GetProgramInfoLog(p, l, nil, gl.Str(logstr))
		log.Fatalln("link:", logstr)
	}
	return p
}

func ortho(left, right, bottom, top, near, far float32) [16]float32 {
	rl := right - left
	tb := top - bottom
	fn := far - near
	return [16]float32{
		2.0 / rl, 0, 0, 0,
		0, 2.0 / tb, 0, 0,
		0, 0, -2.0 / fn, 0,
		-(right + left) / rl, -(top + bottom) / tb, -(far + near) / fn, 1,
	}
}

// Rasterize a single rune into an alpha buffer using font.BoundString for tight bounds
// Replace existing RasterizeRune with this implementation
// This is simpler and more robust: it uses ascent+descent for height and
// places the Drawer.Dot at pad + ascent which reliably renders glyph pixels.
func RasterizeRune(face font.Face, r rune, pad int) ([]uint8, int, int, int, int, int) {
	// Metrics-based approach (robust)
	metrics := face.Metrics()
	ascent := metrics.Ascent.Ceil()
	descent := metrics.Descent.Ceil()
	height := ascent + descent

	// Measure advance
	adv26, _ := face.GlyphAdvance(r)
	adv := (int(adv26) + 63) / 64
	if adv < 1 {
		adv = 8
	}

	// width: use advance as baseline; ensure minimum width
	width := adv
	if width < 8 {
		width = 8
	}

	W := width + pad*2
	H := height + pad*2

	// Create alpha image and clear
	img := image.NewAlpha(image.Rect(0, 0, W, H))
	// Prepare drawer: place Dot so that glyph baseline is at pad + ascent
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.Alpha{255}),
		Face: face,
		Dot: fixed.Point26_6{
			X: fixed.I(pad),
			Y: fixed.I(pad + ascent),
		},
	}
	d.DrawString(string(r))

	// Copy alpha into []uint8
	pix := make([]uint8, W*H)
	for yy := 0; yy < H; yy++ {
		for xx := 0; xx < W; xx++ {
			pix[yy*W+xx] = img.AlphaAt(xx, yy).A
		}
	}

	// Bearings: keep simple and consistent
	bearingX := 0
	bearingY := ascent

	return pix, W, H, adv, bearingX, bearingY
}

func main() {
	flag.StringVar(&fontPath, "font", "./font.ttf", "Path to TTF font file")
	flag.IntVar(&windowWidth, "width", windowWidth, "Window width")
	flag.IntVar(&windowHeight, "height", windowHeight, "Window height")
	flag.IntVar(&fontPixelEm, "size", fontPixelEm, "Font pixel size")
	flag.BoolVar(&useVSync, "vsync", useVSync, "Enable vsync")
	flag.Parse()

	if _, err := os.Stat(fontPath); os.IsNotExist(err) {
		log.Fatalln("font not found:", fontPath)
	}

	initGLFW := func() {
		if err := glfw.Init(); err != nil {
			log.Fatalln("glfw init:", err)
		}
	}
	initGL := func() {
		if err := gl.Init(); err != nil {
			log.Fatalln("gl init:", err)
		}
	}

	initGLFW()
	defer glfw.Terminate()
	glfw.WindowHint(glfw.ContextVersionMajor, 3)
	glfw.WindowHint(glfw.ContextVersionMinor, 3)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	win, err := glfw.CreateWindow(windowWidth, windowHeight, "Font Renderer - Optimized", nil, nil)
	if err != nil {
		log.Fatalln("create window:", err)
	}
	win.MakeContextCurrent()
	if useVSync {
		glfw.SwapInterval(1)
	} else {
		glfw.SwapInterval(0)
	}

	initGL()
	log.Println("GL version:", gl.GoStr(gl.GetString(gl.VERSION)))

	fontBytes, err := os.ReadFile(fontPath)
	if err != nil {
		log.Fatalln("read font:", err)
	}
	tt, err := truetype.Parse(fontBytes)
	if err != nil {
		log.Fatalln("parse ttf:", err)
	}
	opts := &truetype.Options{Size: float64(fontPixelEm), DPI: float64(fontDPI), Hinting: font.HintingFull}
	face := truetype.NewFace(tt, opts)
	defer face.Close()

	// Build runes list
	runes := make([]rune, 0, 95)
	for r := rune(32); r <= 126; r++ {
		runes = append(runes, r)
	}

	// Rasterize glyphs and store
	glyphs := make(map[rune]*Glyph)
	type glyphTemp struct {
		r                 rune
		pix               []uint8
		w, h, adv, bx, by int
	}
	temps := make([]glyphTemp, 0, len(runes))
	for _, r := range runes {
		pix, w, h, adv, bx, by := RasterizeRune(face, r, padding)
		temps = append(temps, glyphTemp{r, pix, w, h, adv, bx, by})
	}

	// Sort by height desc to improve packing
	sort.Slice(temps, func(i, j int) bool { return temps[i].h > temps[j].h })

	packer := NewShelfPacker(atlasWidth, atlasHeight)
	atlas := NewGLAtlas(atlasWidth, atlasHeight)
	packed := 0
	for _, t := range temps {
		x, y, ok := packer.Pack(t.w, t.h)
		if !ok {
			log.Printf("atlas full; skip rune %q", t.r)
			continue
		}
		atlas.UploadSubImage(x, y, t.w, t.h, t.pix)
		g := &Glyph{Rune: t.r, W: t.w, H: t.h, AtlasX: x, AtlasY: y, Advance: t.adv, BearingX: t.bx, BearingY: t.by}
		g.U1 = float32(x) / float32(atlas.W)
		g.V1 = float32(y) / float32(atlas.H)
		g.U2 = float32(x+t.w) / float32(atlas.W)
		g.V2 = float32(y+t.h) / float32(atlas.H)
		glyphs[t.r] = g
		packed++
	}
	log.Printf("packed %d glyphs", packed)
	if err := atlas.DumpPNG("atlas_debug.png"); err != nil {
		log.Println("dump atlas failed:", err)
	}

	// compile shaders
	vs := compileShader(vertexShaderSrc, gl.VERTEX_SHADER)
	fs := compileShader(fragmentShaderSrc, gl.FRAGMENT_SHADER)
	prog := linkProgram(vs, fs)
	gl.DeleteShader(vs)
	gl.DeleteShader(fs)
	gl.UseProgram(prog)

	// Setup VAO/VBO
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	// allocate a large buffer once (bytes)
	initialBytes := int32(4 * 4 * 65536) // 4 floats *4 bytes * 65536 vertices
	gl.BufferData(gl.ARRAY_BUFFER, int(initialBytes), nil, gl.DYNAMIC_DRAW)
	stride := int32(4 * 4) // 4 floats * 4 bytes
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, stride, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 2, gl.FLOAT, false, stride, gl.PtrOffset(2*4))
	gl.BindVertexArray(0)

	projLoc := gl.GetUniformLocation(prog, gl.Str("uProj\x00"))
	colorLoc := gl.GetUniformLocation(prog, gl.Str("uColor\x00"))
	texLoc := gl.GetUniformLocation(prog, gl.Str("uTex\x00"))
	gl.Uniform1i(texLoc, 0)

	proj := ortho(0, float32(windowWidth), float32(windowHeight), 0, -1, 1)
	gl.UniformMatrix4fv(projLoc, 1, false, &proj[0])

	// enable blending
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

	batch := NewBatchTextRenderer(prog, vao, vbo, glyphs, atlas, projLoc, colorLoc)
	fps := NewFPSCounter()

	start := time.Now()
	frame := 0

	for !win.ShouldClose() {
		glfw.PollEvents()
		gl.ClearColor(0.08, 0.08, 0.08, 1.0)
		gl.Clear(gl.COLOR_BUFFER_BIT)

		fps.Update()
		f := fps.GetFPS()

		elapsed := time.Since(start).Seconds()
		wave := math.Sin(elapsed*2) * 50
		pulse := 0.5 + 0.5*math.Sin(elapsed*3)

		// Queue static and dynamic text (avoid fmt for static)
		batch.Draw("Batch Font Renderer - Optimized", 50, 50, [3]float32{1.0, 0.6, 0.2})
		batch.Draw(fmt.Sprintf("FPS: %.1f (VSync:%v)", f, useVSync), 50, 90, [3]float32{0.2, 0.8, 1.0})
		batch.Draw(fmt.Sprintf("Frame: %d", frame), 50, 120, [3]float32{0.8, 0.8, 0.8})
		batch.Draw("0123456789", 50, 210, [3]float32{0.3, 0.9, 0.3})
		batch.Draw("abcdefghijklmnopqrstuvwxyz", 50, 270, [3]float32{0.9, 0.9, 0.3})
		batch.Draw("Waving Text!", 300+int(wave), 350, [3]float32{1.0, float32(pulse * 0.8), 0.2})

		// performance lines
		for i := 0; i < 5; i++ {
			y := 420 + i*20
			col := [3]float32{float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5)), float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5+2.0)), float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5+4.0))}
			batch.Draw(fmt.Sprintf("Performance line %d: ABCDEFGHIJKLMNOPQRSTUVWXYZ", i), 50, y, col)
		}

		batch.Flush()

		win.SwapBuffers()
		frame++
	}
}
