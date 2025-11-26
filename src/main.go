// main_optimized_final.go - Fully optimized batch font renderer
// Applied optimizations:
// - Correct VBO allocation and vertex count handling
// - Group by Color struct (no string formatting / parsing)
// - Preallocated buffers and reduced per-frame allocations
// - Kerning support when available from face.Kern
// - Glyph packing (shelf) with height-sorted glyphs
// - Texture swizzle so alpha lives in texture alpha channel
// - BufferSubData uploads only used portion
// - Stable draw order (keys sorted) to reduce driver state thrash

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

	atlasWidth  = 1024
	atlasHeight = 512

	fontPath    string
	fontDPI     = 72
	fontPixelEm = 32
	padding     = 1

	useVSync = true
)

// FPSCounter
type FPSCounter struct {
	frames int
	last   time.Time
	fps    float64
}

func NewFPSCounter() *FPSCounter { return &FPSCounter{last: time.Now()} }
func (f *FPSCounter) Update() {
	f.frames++
	if time.Since(f.last) >= time.Second {
		f.fps = float64(f.frames) / time.Since(f.last).Seconds()
		f.frames = 0
		f.last = time.Now()
	}
}
func (f *FPSCounter) GetFPS() float64 { return f.fps }

// Glyph holds metadata
type Glyph struct {
	Rune     rune
	W, H     int
	AtlasX   int
	AtlasY   int
	Advance  int
	BearingX int
	BearingY int
	U1, V1   float32
	U2, V2   float32
	Pixels   []uint8
}

// Simple shelf packer
type ShelfPacker struct {
	W, H       int
	x, y, rowH int
}

func NewShelfPacker(w, h int) *ShelfPacker { return &ShelfPacker{W: w, H: h} }
func (s *ShelfPacker) Pack(wid, hei int) (int, int, bool) {
	if wid > s.W || hei > s.H {
		return -1, -1, false
	}
	if s.x+wid > s.W {
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

// GL atlas
type GLAtlas struct {
	Tex  uint32
	W, H int
	CPU  []uint8
}

func NewGLAtlas(w, h int) *GLAtlas {
	var tex uint32
	gl.GenTextures(1, &tex)
	gl.BindTexture(gl.TEXTURE_2D, tex)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.R8, int32(w), int32(h), 0, gl.RED, gl.UNSIGNED_BYTE, nil)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	// swizzle: put red channel into alpha so sampling .a gives glyph alpha
	gl.TexParameteriv(gl.TEXTURE_2D, gl.TEXTURE_SWIZZLE_R, &[]int32{gl.ZERO}[0])
	gl.TexParameteriv(gl.TEXTURE_2D, gl.TEXTURE_SWIZZLE_G, &[]int32{gl.ZERO}[0])
	gl.TexParameteriv(gl.TEXTURE_2D, gl.TEXTURE_SWIZZLE_B, &[]int32{gl.ZERO}[0])
	gl.TexParameteriv(gl.TEXTURE_2D, gl.TEXTURE_SWIZZLE_A, &[]int32{gl.RED}[0])
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

// Vertex layout: pos.xy, uv.xy -> 4 floats

// Color key for grouping
type ColorKey struct{ R, G, B float32 }

func (c ColorKey) Less(o ColorKey) bool {
	if c.R != o.R {
		return c.R < o.R
	}
	if c.G != o.G {
		return c.G < o.G
	}
	return c.B < o.B
}

// Text command
type TextCommand struct {
	Text string
	X, Y int
	C    ColorKey
}

// Batch renderer
type BatchTextRenderer struct {
	prog        uint32
	vao, vbo    uint32
	glyphs      map[rune]*Glyph
	atlas       *GLAtlas
	commands    []TextCommand
	vertices    []float32
	maxVertices int
	colorLoc    int32
	drawCalls   int // Added: counter for draw calls per flush
}

func NewBatchTextRenderer(prog, vao, vbo uint32, glyphs map[rune]*Glyph, atlas *GLAtlas, colorLoc int32) *BatchTextRenderer {
	b := &BatchTextRenderer{prog: prog, vao: vao, vbo: vbo, glyphs: glyphs, atlas: atlas, commands: make([]TextCommand, 0, 256), maxVertices: 65536, colorLoc: colorLoc}
	b.vertices = make([]float32, 0, b.maxVertices*4)
	return b
}
func (b *BatchTextRenderer) Draw(text string, x, y int, c ColorKey) {
	b.commands = append(b.commands, TextCommand{Text: text, X: x, Y: y, C: c})
}

func (b *BatchTextRenderer) Flush(face font.Face) int {
	b.drawCalls = 0 // Reset draw call counter at start of flush
	if len(b.commands) == 0 {
		return 0
	}
	// group by color
	groups := make(map[ColorKey][]TextCommand)
	keys := make([]ColorKey, 0, 16)
	for _, cmd := range b.commands {
		k := cmd.C
		if _, ok := groups[k]; !ok {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], cmd)
	}
	// stable order: sort keys
	sort.Slice(keys, func(i, j int) bool { return keys[i].Less(keys[j]) })

	gl.BindVertexArray(b.vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, b.vbo)
	gl.ActiveTexture(gl.TEXTURE0)
	gl.BindTexture(gl.TEXTURE_2D, b.atlas.Tex)

	for _, k := range keys {
		cmds := groups[k]
		b.vertices = b.vertices[:0]
		for _, cmd := range cmds {
			x := float32(cmd.X)
			y := float32(cmd.Y)
			var prev rune = 0
			for _, ch := range cmd.Text {
				g, ok := b.glyphs[ch]
				if !ok {
					if sp, sok := b.glyphs[' ']; sok {
						x += float32(sp.Advance)
					} else {
						x += 8
					}
					prev = ch
					continue
				}
				// kerning if face provides
				if prev != 0 {
					if kf, ok := face.(interface {
						Kern(r0, r1 rune) fixed.Int26_6
					}); ok {
						kern := kf.Kern(prev, ch)
						x += float32((int(kern) + 63) / 64)
					}
				}
				x0 := x + float32(g.BearingX)
				y0 := y - float32(g.BearingY)
				x1 := x0 + float32(g.W)
				y1 := y0 + float32(g.H)
				u1 := g.U1
				v1 := g.V1
				u2 := g.U2
				v2 := g.V2
				b.vertices = append(b.vertices,
					x0, y0, u1, v1,
					x1, y0, u2, v1,
					x1, y1, u2, v2,
					x0, y0, u1, v1,
					x1, y1, u2, v2,
					x0, y1, u1, v2,
				)
				x += float32(g.Advance)
				prev = ch
			}
		}
		if len(b.vertices) == 0 {
			continue
		}
		// prevent overflow
		if len(b.vertices)/4 > b.maxVertices {
			log.Fatalln("vertex overflow")
		}
		bytes := int(len(b.vertices) * 4)
		gl.BufferSubData(gl.ARRAY_BUFFER, 0, bytes, gl.Ptr(b.vertices))
		gl.Uniform3f(b.colorLoc, k.R, k.G, k.B)
		count := int32(len(b.vertices) / 4)
		gl.DrawArrays(gl.TRIANGLES, 0, count)
		b.drawCalls++ // Count each color group as one draw call
	}
	gl.BindVertexArray(0)
	b.commands = b.commands[:0]
	return b.drawCalls
}

const vertexSrc = `#version 330 core
layout(location=0) in vec2 inPos;
layout(location=1) in vec2 inUV;
out vec2 vUV;
uniform mat4 uProj;
void main(){ vUV = inUV; gl_Position = uProj * vec4(inPos, 0.0, 1.0); }
` + "\x00"
const fragmentSrc = `#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
uniform vec3 uColor;
out vec4 FragColor;
void main(){ float a = texture(uTex, vUV).a; FragColor = vec4(uColor, a); }
` + "\x00"

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
	return [16]float32{2.0 / rl, 0, 0, 0, 0, 2.0 / tb, 0, 0, 0, 0, -2.0 / fn, 0, -(right + left) / rl, -(top + bottom) / tb, -(far + near) / fn, 1}
}

// Robust rasterizer similar to original working approach
func RasterizeRune(face font.Face, r rune, pad int) ([]uint8, int, int, int, int, int) {
	metrics := face.Metrics()
	ascent := metrics.Ascent.Ceil()
	descent := metrics.Descent.Ceil()
	height := ascent + descent
	adv26, _ := face.GlyphAdvance(r)
	adv := (int(adv26) + 63) / 64
	if adv < 1 {
		adv = 8
	}
	width := adv
	if width < 8 {
		width = 8
	}
	W := width + pad*2
	H := height + pad*2
	img := image.NewAlpha(image.Rect(0, 0, W, H))
	d := &font.Drawer{Dst: img, Src: image.NewUniform(color.Alpha{255}), Face: face, Dot: fixed.Point26_6{X: fixed.I(pad), Y: fixed.I(pad + ascent)}}
	d.DrawString(string(r))
	pix := make([]uint8, W*H)
	for yy := 0; yy < H; yy++ {
		for xx := 0; xx < W; xx++ {
			pix[yy*W+xx] = img.AlphaAt(xx, xx).A /* bug-safe fallback below replaced after loop */
		}
	} // correct copy (fixed bug above)
	for yy := 0; yy < H; yy++ {
		for xx := 0; xx < W; xx++ {
			pix[yy*W+xx] = img.AlphaAt(xx, yy).A
		}
	}
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

	if err := glfw.Init(); err != nil {
		log.Fatalln("glfw init:", err)
	}
	defer glfw.Terminate()
	glfw.WindowHint(glfw.ContextVersionMajor, 3)
	glfw.WindowHint(glfw.ContextVersionMinor, 3)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)
	win, err := glfw.CreateWindow(windowWidth, windowHeight, "Font Renderer - Final Optimized", nil, nil)
	if err != nil {
		log.Fatalln("create window:", err)
	}
	win.MakeContextCurrent()
	if useVSync {
		glfw.SwapInterval(1)
	} else {
		glfw.SwapInterval(0)
	}
	if err := gl.Init(); err != nil {
		log.Fatalln("gl init:", err)
	}
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

	// Build runes
	runes := make([]rune, 0, 95)
	for r := rune(32); r <= 126; r++ {
		runes = append(runes, r)
	}

	// Rasterize temps
	type temp struct {
		r                 rune
		pix               []uint8
		w, h, adv, bx, by int
	}
	temps := make([]temp, 0, len(runes))
	for _, r := range runes {
		pix, w, h, adv, bx, by := RasterizeRune(face, r, padding)
		temps = append(temps, temp{r, pix, w, h, adv, bx, by})
	}

	// sort by height desc
	sort.Slice(temps, func(i, j int) bool { return temps[i].h > temps[j].h })

	packer := NewShelfPacker(atlasWidth, atlasHeight)
	atlas := NewGLAtlas(atlasWidth, atlasHeight)
	glyphs := make(map[rune]*Glyph)
	packed := 0
	for _, t := range temps {
		x, y, ok := packer.Pack(t.w, t.h)
		if !ok {
			log.Printf("atlas full; skip %q", t.r)
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

	// shaders
	vs := compileShader(vertexSrc, gl.VERTEX_SHADER)
	fs := compileShader(fragmentSrc, gl.FRAGMENT_SHADER)
	prog := linkProgram(vs, fs)
	gl.DeleteShader(vs)
	gl.DeleteShader(fs)
	gl.UseProgram(prog)

	// VAO/VBO
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	maxVertices := 65536
	initialBytes := int32(maxVertices * 4 * 4) // vertices * floats * bytes
	gl.BufferData(gl.ARRAY_BUFFER, int(initialBytes), nil, gl.DYNAMIC_DRAW)
	stride := int32(4 * 4)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, stride, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 2, gl.FLOAT, false, stride, gl.PtrOffset(2*4))
	gl.BindVertexArray(0)

	colorLoc := gl.GetUniformLocation(prog, gl.Str("uColor\x00"))
	texLoc := gl.GetUniformLocation(prog, gl.Str("uTex\x00"))
	gl.Uniform1i(texLoc, 0)
	projLoc := gl.GetUniformLocation(prog, gl.Str("uProj\x00"))
	proj := ortho(0, float32(windowWidth), float32(windowHeight), 0, -1, 1)
	gl.UniformMatrix4fv(projLoc, 1, false, &proj[0])

	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

	batch := NewBatchTextRenderer(prog, vao, vbo, glyphs, atlas, colorLoc)
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

		batch.Draw("Batch Font Renderer - Final", 50, 50, ColorKey{1.0, 0.6, 0.2})
		batch.Draw(fmt.Sprintf("FPS: %.1f (VSync:%v)", f, useVSync), 50, 90, ColorKey{0.2, 0.8, 1.0})
		batch.Draw(fmt.Sprintf("Frame: %d", frame), 50, 120, ColorKey{0.8, 0.8, 0.8})
		batch.Draw("0123456789", 50, 210, ColorKey{0.3, 0.9, 0.3})
		batch.Draw("abcdefghijklmnopqrstuvwxyz", 50, 270, ColorKey{0.9, 0.9, 0.3})
		batch.Draw("Waving Text!", 300+int(wave), 350, ColorKey{1.0, float32(pulse * 0.8), 0.2})
		for i := 0; i < 8; i++ {
			y := 420 + i*20
			col := ColorKey{float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5)), float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5+2.0)), float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5+4.0))}
			batch.Draw(fmt.Sprintf("Performance line %d: ABCDEFGHIJKLMNOPQRSTUVWXYZ", i), 50, y, col)
		}

		drawCalls := batch.Flush(face) // Get draw calls count for this frame

		// Draw the draw calls info on screen
		batch.Draw(fmt.Sprintf("Draw Calls: %d", drawCalls), 50, 150, ColorKey{0.9, 0.5, 0.9})
		batch.Flush(face) // Flush the draw calls info text

		win.SwapBuffers()
		frame++
	}
}
