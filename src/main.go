// main.go - Fixed batch rendering with colors
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"math"
	"os"
	"runtime"
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
	atlasHeight = 512

	fontPath    string
	fontDPI     = 72
	fontPixelEm = 32
	padding     = 2

	useVSync = true // Enable VSync by default
)

// FPS counter
type FPSCounter struct {
	frames   int
	lastTime time.Time
	fps      float64
}

func NewFPSCounter() *FPSCounter {
	return &FPSCounter{
		frames:   0,
		lastTime: time.Now(),
		fps:      0,
	}
}

func (f *FPSCounter) Update() {
	f.frames++
	if time.Since(f.lastTime) >= time.Second {
		f.fps = float64(f.frames) / time.Since(f.lastTime).Seconds()
		f.frames = 0
		f.lastTime = time.Now()
	}
}

func (f *FPSCounter) GetFPS() float64 {
	return f.fps
}

// TextCommand represents a text rendering command
type TextCommand struct {
	text  string
	x, y  int
	color [3]float32
}

// BatchTextRenderer handles batched text drawing
type BatchTextRenderer struct {
	prog   uint32
	vao    uint32
	vbo    uint32
	glyphs map[rune]*Glyph
	atlas  *GLAtlas

	// Batch rendering fields
	vertices []float32
	commands []TextCommand
}

func NewBatchTextRenderer(prog, vao, vbo uint32, glyphs map[rune]*Glyph, atlas *GLAtlas) *BatchTextRenderer {
	return &BatchTextRenderer{
		prog:     prog,
		vao:      vao,
		vbo:      vbo,
		glyphs:   glyphs,
		atlas:    atlas,
		vertices: make([]float32, 0, 1024), // Pre-allocate capacity
		commands: make([]TextCommand, 0, 64),
	}
}

// Draw queues text for rendering (doesn't render immediately)
func (b *BatchTextRenderer) Draw(text string, x, y int, color [3]float32) {
	b.commands = append(b.commands, TextCommand{
		text:  text,
		x:     x,
		y:     y,
		color: color,
	})
}

// Flush renders all queued text in one batch
func (b *BatchTextRenderer) Flush() {
	if len(b.commands) == 0 {
		return
	}

	// Reset vertices but keep underlying array
	b.vertices = b.vertices[:0]

	// Build one big vertex buffer for all text
	for _, cmd := range b.commands {
		b.appendTextVertices(cmd.text, cmd.x, cmd.y, cmd.color)
	}

	// Single draw call for all text
	if len(b.vertices) > 0 {
		gl.BindVertexArray(b.vao)
		gl.BindBuffer(gl.ARRAY_BUFFER, b.vbo)
		gl.BufferData(gl.ARRAY_BUFFER, len(b.vertices)*4, gl.Ptr(b.vertices), gl.DYNAMIC_DRAW)
		cnt := int32(len(b.vertices) / 4)
		gl.DrawArrays(gl.TRIANGLES, 0, cnt)
		gl.BindVertexArray(0)
	}

	// Clear commands for next frame
	b.commands = b.commands[:0]
}

// GetVertexCount returns the number of vertices in the current batch (for debugging)
func (b *BatchTextRenderer) GetVertexCount() int {
	return len(b.vertices)
}

// GetCommandCount returns the number of queued text commands (for debugging)
func (b *BatchTextRenderer) GetCommandCount() int {
	return len(b.commands)
}

func (b *BatchTextRenderer) appendTextVertices(s string, startX, startY int, textColor [3]float32) {
	x := float32(startX)
	y := float32(startY)

	for _, ch := range s {
		g, ok := b.glyphs[ch]
		if !ok {
			// Use space width as fallback
			if space, ok := b.glyphs[' ']; ok {
				x += float32(space.Advance)
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

		// Append vertices to the batch (6 vertices per character = 2 triangles)
		// Each vertex now includes color data (3 floats for RGB)
		b.vertices = append(b.vertices,
			// Triangle 1 - Vertex 1
			x0, y0, u1, v1, textColor[0], textColor[1], textColor[2],
			// Triangle 1 - Vertex 2
			x1, y0, u2, v1, textColor[0], textColor[1], textColor[2],
			// Triangle 1 - Vertex 3
			x1, y1, u2, v2, textColor[0], textColor[1], textColor[2],
			// Triangle 2 - Vertex 1
			x0, y0, u1, v1, textColor[0], textColor[1], textColor[2],
			// Triangle 2 - Vertex 2
			x1, y1, u2, v2, textColor[0], textColor[1], textColor[2],
			// Triangle 2 - Vertex 3
			x0, y1, u1, v2, textColor[0], textColor[1], textColor[2],
		)

		x += float32(g.Advance)
	}
}

// SimpleAtlas packer
type SimpleAtlas struct {
	W, H int
	x, y int
	rowH int
}

func NewSimpleAtlas(w, h int) *SimpleAtlas {
	return &SimpleAtlas{
		W:    w,
		H:    h,
		x:    0,
		y:    0,
		rowH: 0,
	}
}

func (a *SimpleAtlas) Pack(wid, hei int) (int, int) {
	// If this glyph doesn't fit in current row, move to next row
	if a.x+wid > a.W {
		a.x = 0
		a.y += a.rowH
		a.rowH = 0
	}

	// If we're out of space, return failure
	if a.y+hei > a.H {
		return -1, -1
	}

	// Update row height if needed
	if hei > a.rowH {
		a.rowH = hei
	}

	// Get position and advance x
	x, y := a.x, a.y
	a.x += wid

	return x, y
}

// GL Atlas
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
	return &GLAtlas{Tex: tex, W: w, H: h, CPU: make([]uint8, w*h)}
}

func (a *GLAtlas) UploadSubImage(x, y, w, h int, pixels []uint8) {
	if len(pixels) < w*h {
		log.Println("UploadSubImage: pixels slice too small")
		return
	}
	for row := 0; row < h; row++ {
		dstOff := (y+row)*a.W + x
		srcOff := row * w
		copy(a.CPU[dstOff:dstOff+w], pixels[srcOff:srcOff+w])
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

// Glyph struct
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
}

// RasterizeGlyphFace
func RasterizeGlyphFace(face font.Face, r rune, pad int) ([]uint8, int, int, int, int, int) {
	// Get metrics
	metrics := face.Metrics()
	ascent := metrics.Ascent.Ceil()
	height := ascent + metrics.Descent.Ceil()

	// Measure advance
	advance, _ := face.GlyphAdvance(r)
	adv := (int(advance) + 32) / 64
	if adv < 1 {
		adv = 8
	}

	// Simple width estimation
	width := adv
	if width < 8 {
		width = 8
	}

	// Add padding
	W := width + pad*2
	H := height + pad*2

	// Create image and draw
	img := image.NewAlpha(image.Rect(0, 0, W, H))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.Alpha{0}}, image.Point{}, draw.Src)

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

	// Convert to []uint8
	pix := make([]uint8, W*H)
	for yy := 0; yy < H; yy++ {
		for xx := 0; xx < W; xx++ {
			pix[yy*W+xx] = img.AlphaAt(xx, yy).A
		}
	}

	// Simple bearings
	bearingX := 0
	bearingY := ascent

	return pix, W, H, adv, bearingX, bearingY
}

// Updated shader sources with color in vertex attributes
const vertexShader = `#version 330 core
layout(location=0) in vec2 inPos;
layout(location=1) in vec2 inUV;
layout(location=2) in vec3 inColor;
out vec2 vUV;
out vec3 vColor;
uniform mat4 uProj;
void main() {
    vUV = inUV;
    vColor = inColor;
    gl_Position = uProj * vec4(inPos,0.0,1.0);
}` + "\x00"

const fragmentShader = `#version 330 core
in vec2 vUV;
in vec3 vColor;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    float a = texture(uTex, vUV).r;
    FragColor = vec4(vColor, a);
}` + "\x00"

// GL helpers
func init() {
	runtime.LockOSThread()
}

func initGLFW() {
	if err := glfw.Init(); err != nil {
		log.Fatalln("glfw init:", err)
	}
}

func initGL() {
	if err := gl.Init(); err != nil {
		log.Fatalln("gl init:", err)
	}
	log.Println("GL version:", gl.GoStr(gl.GetString(gl.VERSION)))
}

func compileShader(src string, shaderType uint32) uint32 {
	sh := gl.CreateShader(shaderType)
	cstrs, free := gl.Strs(src)
	gl.ShaderSource(sh, 1, cstrs, nil)
	free()
	gl.CompileShader(sh)
	var status int32
	gl.GetShaderiv(sh, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var l int32
		gl.GetShaderiv(sh, gl.INFO_LOG_LENGTH, &l)
		logstr := strings.Repeat("\x00", int(l+1))
		gl.GetShaderInfoLog(sh, l, nil, gl.Str(logstr))
		log.Fatalln("shader compile error:", logstr)
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
		log.Fatalln("link error:", logstr)
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

// ---- main ----
func main() {
	// Parse command line arguments
	flag.StringVar(&fontPath, "font", "./font.ttf", "Path to TTF font file")
	flag.IntVar(&windowWidth, "width", 800, "Window width")
	flag.IntVar(&windowHeight, "height", 600, "Window height")
	flag.IntVar(&fontPixelEm, "size", 32, "Font size in pixels")
	flag.BoolVar(&useVSync, "vsync", true, "Enable VSync")
	flag.Parse()

	if fontPath == "" {
		log.Fatal("Please specify a font file using -font flag")
	}

	// Check if font file exists
	if _, err := os.Stat(fontPath); os.IsNotExist(err) {
		log.Fatalf("Font file not found: %s", fontPath)
	}

	log.Printf("Using font: %s (size: %d, VSync: %v)", fontPath, fontPixelEm, useVSync)

	initGLFW()
	defer glfw.Terminate()

	// Set OpenGL context hints
	glfw.WindowHint(glfw.ContextVersionMajor, 3)
	glfw.WindowHint(glfw.ContextVersionMinor, 3)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	win, err := glfw.CreateWindow(windowWidth, windowHeight, "Font Renderer (Batch) - "+fontPath, nil, nil)
	if err != nil {
		log.Fatalln("create window:", err)
	}
	win.MakeContextCurrent()

	// Set VSync
	if useVSync {
		glfw.SwapInterval(1)
		log.Println("VSync enabled")
	} else {
		glfw.SwapInterval(0)
		log.Println("VSync disabled")
	}

	initGL()

	// Load TTF
	fontBytes, err := os.ReadFile(fontPath)
	if err != nil {
		log.Fatalf("read font %s: %v", fontPath, err)
	}
	tt, err := truetype.Parse(fontBytes)
	if err != nil {
		log.Fatalln("parse ttf:", err)
	}
	opts := &truetype.Options{Size: float64(fontPixelEm), DPI: float64(fontDPI), Hinting: font.HintingFull}
	face := truetype.NewFace(tt, opts)
	defer face.Close()

	// Create atlas
	atlas := NewSimpleAtlas(atlasWidth, atlasHeight)
	glAtlas := NewGLAtlas(atlasWidth, atlasHeight)

	// Build ASCII glyphs
	glyphs := make(map[rune]*Glyph)
	runes := []rune{}
	for r := rune(32); r <= 126; r++ {
		runes = append(runes, r)
	}

	log.Println("Packing glyphs into atlas...")
	for _, r := range runes {
		pix, w, h, adv, bx, by := RasterizeGlyphFace(face, r, padding)
		x, y := atlas.Pack(w, h)
		if x == -1 {
			log.Printf("atlas full, cannot pack glyph %q, skipping", r)
			continue
		}
		glAtlas.UploadSubImage(x, y, w, h, pix)
		g := &Glyph{
			Rune:     r,
			W:        w,
			H:        h,
			AtlasX:   x,
			AtlasY:   y,
			Advance:  adv,
			BearingX: bx,
			BearingY: by,
		}
		g.U1 = float32(x) / float32(atlasWidth)
		g.V1 = float32(y) / float32(atlasHeight)
		g.U2 = float32(x+w) / float32(atlasWidth)
		g.V2 = float32(y+h) / float32(atlasHeight)
		glyphs[r] = g
	}
	log.Printf("Successfully packed %d glyphs into atlas", len(glyphs))

	// Write debug PNG
	if err := glAtlas.DumpPNG("atlas_debug.png"); err != nil {
		log.Println("dump atlas png failed:", err)
	} else {
		log.Println("wrote atlas_debug.png")
	}

	// Compile shader
	vs := compileShader(vertexShader, gl.VERTEX_SHADER)
	fs := compileShader(fragmentShader, gl.FRAGMENT_SHADER)
	prog := linkProgram(vs, fs)
	gl.DeleteShader(vs)
	gl.DeleteShader(fs)

	// VAO/VBO - UPDATED for color attributes
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	// Pre-allocate a larger buffer for batch rendering
	initialBufferSize := 65536 * 2 // Larger buffer for color data
	gl.BufferData(gl.ARRAY_BUFFER, initialBufferSize, nil, gl.DYNAMIC_DRAW)

	// Updated vertex attribute layout: position(2) + uv(2) + color(3) = 7 floats per vertex
	stride := int32(7 * 4) // 7 floats * 4 bytes each
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, stride, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 2, gl.FLOAT, false, stride, gl.PtrOffset(2*4))
	gl.EnableVertexAttribArray(2)
	gl.VertexAttribPointer(2, 3, gl.FLOAT, false, stride, gl.PtrOffset(4*4))
	gl.BindVertexArray(0)

	// Projection
	gl.UseProgram(prog)
	projLoc := gl.GetUniformLocation(prog, gl.Str("uProj\x00"))
	proj := ortho(0, float32(windowWidth), float32(windowHeight), 0, -1, 1)
	gl.UniformMatrix4fv(projLoc, 1, false, &proj[0])

	// Enable blending
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

	// Create batch text renderer
	batchRenderer := NewBatchTextRenderer(prog, vao, vbo, glyphs, glAtlas)

	// Create FPS counter
	fpsCounter := NewFPSCounter()

	// Animation variables
	startTime := time.Now()
	frameCount := 0

	log.Println("Starting batch rendering loop...")

	for !win.ShouldClose() {
		glfw.PollEvents()
		gl.ClearColor(0.1, 0.1, 0.1, 1.0)
		gl.Clear(gl.COLOR_BUFFER_BIT)

		// Update FPS counter
		fpsCounter.Update()
		fps := fpsCounter.GetFPS()

		// Update animation
		elapsed := time.Since(startTime).Seconds()
		waveOffset := math.Sin(elapsed*2) * 50
		pulse := 0.5 + 0.5*math.Sin(elapsed*3)

		// Set up rendering state (once per frame)
		gl.UseProgram(prog)
		gl.ActiveTexture(gl.TEXTURE0)
		gl.BindTexture(gl.TEXTURE_2D, glAtlas.Tex)

		// Queue all text for this frame (no rendering happens yet)
		batchRenderer.Draw("Batch Font Renderer Demo", 50, 50, [3]float32{1.0, 0.5, 0.2})
		batchRenderer.Draw(fmt.Sprintf("FPS: %.1f (VSync: %v)", fps, useVSync), 50, 90, [3]float32{0.2, 0.8, 1.0})
		batchRenderer.Draw(fmt.Sprintf("Frame: %d", frameCount), 50, 120, [3]float32{0.7, 0.7, 0.7})
		batchRenderer.Draw(fmt.Sprintf("Commands: %d, Vertices: %d",
			batchRenderer.GetCommandCount(), batchRenderer.GetVertexCount()), 50, 150, [3]float32{0.7, 0.7, 0.7})
		batchRenderer.Draw("Font: "+fontPath, 50, 180, [3]float32{0.8, 0.8, 0.8})
		batchRenderer.Draw("0123456789", 50, 210, [3]float32{0.3, 0.9, 0.3})
		batchRenderer.Draw("Characters: ABCDEFGHIJKLMNOPQRSTUVWXYZ", 50, 240, [3]float32{0.9, 0.9, 0.3})
		batchRenderer.Draw("abcdefghijklmnopqrstuvwxyz", 50, 270, [3]float32{0.9, 0.9, 0.3})
		batchRenderer.Draw("!@#$%^&*()_+-=[]{}|;:,.<>?/~", 50, 300, [3]float32{0.9, 0.3, 0.3})

		// Animated text
		textColor := [3]float32{1.0, float32(pulse * 0.8), 0.2}
		batchRenderer.Draw("Waving Text!", 300+int(waveOffset), 350, textColor)

		// Colorful text - these should now show different colors!
		batchRenderer.Draw("R", 50, 380, [3]float32{1.0, 0.0, 0.0})
		batchRenderer.Draw("A", 80, 380, [3]float32{1.0, 0.5, 0.0})
		batchRenderer.Draw("I", 110, 380, [3]float32{1.0, 1.0, 0.0})
		batchRenderer.Draw("N", 130, 380, [3]float32{0.0, 1.0, 0.0})
		batchRenderer.Draw("B", 160, 380, [3]float32{0.0, 0.5, 1.0})
		batchRenderer.Draw("O", 190, 380, [3]float32{0.5, 0.0, 1.0})
		batchRenderer.Draw("W", 220, 380, [3]float32{1.0, 0.0, 1.0})

		// Performance test - render many characters to demonstrate batch efficiency
		for i := 0; i < 5; i++ {
			yPos := 420 + i*20
			color := [3]float32{
				float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5)),
				float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5+2.0)),
				float32(0.5 + 0.5*math.Sin(elapsed+float64(i)*0.5+4.0)),
			}
			batchRenderer.Draw(fmt.Sprintf("Performance line %d: ABCDEFGHIJKLMNOPQRSTUVWXYZ", i),
				50, yPos, color)
		}

		// Single render call for ALL text in this frame
		batchRenderer.Flush()

		// Swap buffers (with VSync if enabled)
		win.SwapBuffers()
	}
}
