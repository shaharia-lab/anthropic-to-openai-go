package anthropic2openai

import (
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

func TestTranslateRequestSystemWithParts(t *testing.T) {
	src := &openai.ChatCompletionRequest{
		Model: "glm-4.6",
		Messages: []openai.Message{
			{Role: openai.RoleSystem, Content: openai.Content{Parts: []openai.ContentPart{
				{Type: openai.PartTypeText, Text: "sys"},
				{Type: openai.PartTypeImageURL, ImageURL: &openai.ImageURL{URL: "ignored"}},
			}}},
			{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}},
		},
	}
	got, err := TranslateRequest(src, 0)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if got.System != "sys" {
		t.Fatalf("system = %q (image parts should be ignored in system text)", got.System)
	}
}

func TestUserBlocksErrors(t *testing.T) {
	if _, err := userBlocks(openai.Content{Parts: []openai.ContentPart{{Type: "audio"}}}); err == nil {
		t.Fatal("expected error for unsupported part type")
	}
	if _, err := userBlocks(openai.Content{Parts: []openai.ContentPart{{Type: openai.PartTypeImageURL}}}); err == nil {
		t.Fatal("expected error for image part with no image_url")
	}
}

func TestAssistantBlocksEmptyEmitsTextBlock(t *testing.T) {
	blocks, err := assistantBlocks(openai.Message{Role: openai.RoleAssistant})
	if err != nil {
		t.Fatalf("assistantBlocks: %v", err)
	}
	if len(blocks) != 1 || blocks[0].Text != "" {
		t.Fatalf("expected single empty text block, got %#v", blocks)
	}
}
