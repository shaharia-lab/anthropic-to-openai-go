package anthropic2openai

import (
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
)

func TestImageBlockRemoteURL(t *testing.T) {
	block, err := imageBlock("https://example.com/cat.png")
	if err != nil {
		t.Fatalf("imageBlock: %v", err)
	}
	if block.Type != anthropic.BlockImage || block.Source == nil {
		t.Fatalf("unexpected block: %+v", block)
	}
	if block.Source.Type != anthropic.SourceURL || block.Source.URL != "https://example.com/cat.png" {
		t.Fatalf("unexpected source: %+v", block.Source)
	}
}

func TestImageBlockDataURL(t *testing.T) {
	block, err := imageBlock("data:image/jpeg;base64,aGk=")
	if err != nil {
		t.Fatalf("imageBlock: %v", err)
	}
	if block.Source.Type != anthropic.SourceBase64 {
		t.Fatalf("source type = %q", block.Source.Type)
	}
	if block.Source.MediaType != "image/jpeg" || block.Source.Data != "aGk=" {
		t.Fatalf("unexpected source: %+v", block.Source)
	}
}

func TestImageBlockErrors(t *testing.T) {
	cases := []struct {
		name string
		url  string
	}{
		{"empty", ""},
		{"no comma", "data:image/png;base64"},
		{"not base64 encoding", "data:image/png,raw"},
		{"missing media type", "data:;base64,aGk="},
		{"invalid base64", "data:image/png;base64,!!!!"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := imageBlock(tc.url); err == nil {
				t.Fatalf("expected error for %q", tc.url)
			}
		})
	}
}

func TestImageBlockDataURLWithCharset(t *testing.T) {
	block, err := imageBlock("data:image/webp;charset=utf-8;base64,aGk=")
	if err != nil {
		t.Fatalf("imageBlock: %v", err)
	}
	if block.Source.MediaType != "image/webp" {
		t.Fatalf("media type = %q", block.Source.MediaType)
	}
}
