//go:build integration

package anthropic2openai

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// imageSize is the edge length, in pixels, of the generated test image.
const imageSize = 256

// solidColors maps the color names used in tests to their RGBA values, used by
// the pure-Go fallback renderer.
var solidColors = map[string]color.RGBA{
	"green": {R: 0x2e, G: 0xa0, B: 0x43, A: 0xff},
	"blue":  {R: 0x1e, G: 0x66, B: 0xd0, A: 0xff},
	"red":   {R: 0xd0, G: 0x2e, B: 0x2e, A: 0xff},
}

// solidColorImageDataURL renders a solid-color square as a PNG and returns it as
// an RFC 2397 data URL. It first builds an SVG and rasterises it with
// ImageMagick (proving the data path with a genuinely encoded raster image),
// falling back to a pure-Go PNG when ImageMagick is unavailable.
func solidColorImageDataURL(t *testing.T, colorName string) string {
	t.Helper()
	pngBytes, ok := svgToPNG(t, solidColorSVG(colorName))
	if !ok {
		pngBytes = drawSolidPNG(t, colorName)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
}

// solidColorSVG returns an SVG document filled with the named color.
func solidColorSVG(colorName string) string {
	return fmt.Sprintf(
		`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d"><rect width="%d" height="%d" fill="%s"/></svg>`,
		imageSize, imageSize, imageSize, imageSize, colorName,
	)
}

// svgToPNG rasterises an SVG to PNG using ImageMagick's convert tool, returning
// ok=false when the tool is unavailable.
func svgToPNG(t *testing.T, svg string) ([]byte, bool) {
	t.Helper()
	bin, err := exec.LookPath("convert")
	if err != nil {
		return nil, false
	}
	dir := t.TempDir()
	svgPath := filepath.Join(dir, "image.svg")
	pngPath := filepath.Join(dir, "image.png")
	if err := os.WriteFile(svgPath, []byte(svg), 0o600); err != nil {
		t.Fatalf("write svg: %v", err)
	}
	if out, err := exec.Command(bin, svgPath, pngPath).CombinedOutput(); err != nil {
		t.Logf("imagemagick convert failed (%v): %s", err, out)
		return nil, false
	}
	data, err := os.ReadFile(pngPath)
	if err != nil {
		t.Fatalf("read png: %v", err)
	}
	return data, true
}

// drawSolidPNG produces a solid-color PNG using only the standard library.
func drawSolidPNG(t *testing.T, colorName string) []byte {
	t.Helper()
	c, ok := solidColors[colorName]
	if !ok {
		t.Fatalf("unknown test color %q", colorName)
	}
	img := image.NewRGBA(image.Rect(0, 0, imageSize, imageSize))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: c}, image.Point{}, draw.Src)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}
