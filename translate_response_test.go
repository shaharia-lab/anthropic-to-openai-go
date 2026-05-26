package anthropic2openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

func TestTranslateResponseText(t *testing.T) {
	src := &anthropic.MessagesResponse{
		ID:         "msg_123",
		Model:      "glm-4.6",
		Content:    []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: "Hello "}, {Type: anthropic.BlockText, Text: "world"}},
		StopReason: anthropic.StopEndTurn,
		Usage:      anthropic.Usage{InputTokens: 11, OutputTokens: 3},
	}
	got := TranslateResponse(src, 1700)

	if !strings.HasPrefix(got.ID, completionIDPrefix) || !strings.Contains(got.ID, "msg_123") {
		t.Fatalf("id = %q", got.ID)
	}
	if got.Object != openai.ObjectChatCompletion || got.Created != 1700 {
		t.Fatalf("envelope mismatch: %+v", got)
	}
	choice := got.Choices[0]
	if choice.Message.Content == nil || *choice.Message.Content != "Hello world" {
		t.Fatalf("content = %v", choice.Message.Content)
	}
	if choice.FinishReason != openai.FinishReasonStop {
		t.Fatalf("finish = %q", choice.FinishReason)
	}
	if got.Usage != (openai.Usage{PromptTokens: 11, CompletionTokens: 3, TotalTokens: 14}) {
		t.Fatalf("usage = %+v", got.Usage)
	}
}

func TestTranslateResponseToolUse(t *testing.T) {
	src := &anthropic.MessagesResponse{
		ID:    "msg_456",
		Model: "glm-4.6",
		Content: []anthropic.ContentBlock{
			{Type: anthropic.BlockToolUse, ID: "toolu_1", Name: "get_weather", Input: json.RawMessage(`{"city":"NYC"}`)},
		},
		StopReason: anthropic.StopToolUse,
	}
	got := TranslateResponse(src, 1)
	choice := got.Choices[0]
	if choice.FinishReason != openai.FinishReasonToolCalls {
		t.Fatalf("finish = %q", choice.FinishReason)
	}
	if choice.Message.Content != nil {
		t.Fatalf("expected nil content, got %v", *choice.Message.Content)
	}
	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d", len(choice.Message.ToolCalls))
	}
	call := choice.Message.ToolCalls[0]
	if call.ID != "toolu_1" || call.Type != openai.ToolTypeFunction {
		t.Fatalf("call = %+v", call)
	}
	if call.Function.Name != "get_weather" || call.Function.Arguments != `{"city":"NYC"}` {
		t.Fatalf("function = %+v", call.Function)
	}
}

func TestTranslateResponseTextWithToolUse(t *testing.T) {
	src := &anthropic.MessagesResponse{
		Model: "glm-4.6",
		Content: []anthropic.ContentBlock{
			{Type: anthropic.BlockText, Text: "Let me check"},
			{Type: anthropic.BlockToolUse, ID: "toolu_1", Name: "f", Input: json.RawMessage(`{}`)},
		},
		StopReason: anthropic.StopToolUse,
	}
	got := TranslateResponse(src, 1)
	choice := got.Choices[0]
	if choice.Message.Content == nil || *choice.Message.Content != "Let me check" {
		t.Fatalf("content = %v", choice.Message.Content)
	}
	if choice.FinishReason != openai.FinishReasonToolCalls {
		t.Fatalf("finish = %q", choice.FinishReason)
	}
}

func TestMapStopReason(t *testing.T) {
	cases := []struct {
		reason       string
		hasToolCalls bool
		want         string
	}{
		{anthropic.StopEndTurn, false, openai.FinishReasonStop},
		{anthropic.StopStopSequence, false, openai.FinishReasonStop},
		{anthropic.StopMaxTokens, false, openai.FinishReasonLength},
		{anthropic.StopToolUse, false, openai.FinishReasonToolCalls},
		{anthropic.StopEndTurn, true, openai.FinishReasonToolCalls},
		{"", false, openai.FinishReasonStop},
	}
	for _, tc := range cases {
		if got := mapStopReason(tc.reason, tc.hasToolCalls); got != tc.want {
			t.Fatalf("mapStopReason(%q,%v) = %q want %q", tc.reason, tc.hasToolCalls, got, tc.want)
		}
	}
}

func TestTranslateResponseEmptyToolInputDefaultsToObject(t *testing.T) {
	src := &anthropic.MessagesResponse{
		Content: []anthropic.ContentBlock{{Type: anthropic.BlockToolUse, ID: "t", Name: "f"}},
	}
	got := TranslateResponse(src, 1)
	if got.Choices[0].Message.ToolCalls[0].Function.Arguments != "{}" {
		t.Fatalf("arguments = %q", got.Choices[0].Message.ToolCalls[0].Function.Arguments)
	}
}

func TestCompletionIDGeneratedWhenUpstreamEmpty(t *testing.T) {
	id := completionID("")
	if !strings.HasPrefix(id, completionIDPrefix) {
		t.Fatalf("id = %q", id)
	}
	if len(id) <= len(completionIDPrefix) {
		t.Fatalf("expected generated suffix, got %q", id)
	}
	first, second := completionID(""), completionID("")
	if first == second {
		t.Fatal("generated IDs should not collide")
	}
}
