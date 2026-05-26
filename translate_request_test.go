package anthropic2openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

func TestTranslateRequestNil(t *testing.T) {
	if _, err := TranslateRequest(nil, 0); err == nil {
		t.Fatal("expected error for nil request")
	}
}

func TestTranslateRequestSystemAndMultiTurn(t *testing.T) {
	src := &openai.ChatCompletionRequest{
		Model: "glm-4.6",
		Messages: []openai.Message{
			{Role: openai.RoleSystem, Content: openai.Content{Text: "be brief"}},
			{Role: openai.RoleSystem, Content: openai.Content{Text: "be kind"}},
			{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}},
			{Role: openai.RoleAssistant, Content: openai.Content{Text: "hello"}},
			{Role: openai.RoleUser, Content: openai.Content{Text: "bye"}},
		},
	}
	got, err := TranslateRequest(src, 0)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if got.System != "be brief\n\nbe kind" {
		t.Fatalf("system = %q", got.System)
	}
	want := []anthropic.Message{
		{Role: anthropic.RoleUser, Content: []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: "hi"}}},
		{Role: anthropic.RoleAssistant, Content: []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: "hello"}}},
		{Role: anthropic.RoleUser, Content: []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: "bye"}}},
	}
	if !reflect.DeepEqual(got.Messages, want) {
		t.Fatalf("messages mismatch:\n got %#v\nwant %#v", got.Messages, want)
	}
	if got.MaxTokens != DefaultMaxTokens {
		t.Fatalf("max_tokens = %d", got.MaxTokens)
	}
}

func TestTranslateRequestToolCallRoundTrip(t *testing.T) {
	src := &openai.ChatCompletionRequest{
		Model: "glm-4.6",
		Messages: []openai.Message{
			{Role: openai.RoleUser, Content: openai.Content{Text: "weather?"}},
			{Role: openai.RoleAssistant, ToolCalls: []openai.ToolCall{
				{ID: "call_1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "get_weather", Arguments: `{"city":"NYC"}`}},
				{ID: "call_2", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "get_time", Arguments: ""}},
			}},
			{Role: openai.RoleTool, ToolCallID: "call_1", Content: openai.Content{Text: "sunny"}},
			{Role: openai.RoleTool, ToolCallID: "call_2", Content: openai.Content{Text: "noon"}},
		},
	}
	got, err := TranslateRequest(src, 0)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	want := []anthropic.Message{
		{Role: anthropic.RoleUser, Content: []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: "weather?"}}},
		{Role: anthropic.RoleAssistant, Content: []anthropic.ContentBlock{
			{Type: anthropic.BlockToolUse, ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"NYC"}`)},
			{Type: anthropic.BlockToolUse, ID: "call_2", Name: "get_time", Input: json.RawMessage(`{}`)},
		}},
		{Role: anthropic.RoleUser, Content: []anthropic.ContentBlock{
			{Type: anthropic.BlockToolResult, ToolUseID: "call_1", Content: json.RawMessage(`"sunny"`)},
			{Type: anthropic.BlockToolResult, ToolUseID: "call_2", Content: json.RawMessage(`"noon"`)},
		}},
	}
	if !reflect.DeepEqual(got.Messages, want) {
		t.Fatalf("messages mismatch:\n got %#v\nwant %#v", got.Messages, want)
	}
}

func TestTranslateRequestInvalidToolArguments(t *testing.T) {
	src := &openai.ChatCompletionRequest{
		Model: "glm-4.6",
		Messages: []openai.Message{
			{Role: openai.RoleAssistant, ToolCalls: []openai.ToolCall{
				{ID: "call_1", Function: openai.FunctionCall{Name: "f", Arguments: "{not json"}},
			}},
		},
	}
	if _, err := TranslateRequest(src, 0); err == nil {
		t.Fatal("expected error for invalid tool arguments")
	}
}

func TestTranslateRequestImageParts(t *testing.T) {
	dataURL := "data:image/png;base64,aGVsbG8="
	src := &openai.ChatCompletionRequest{
		Model: "glm-4.6",
		Messages: []openai.Message{{
			Role: openai.RoleUser,
			Content: openai.Content{Parts: []openai.ContentPart{
				{Type: openai.PartTypeText, Text: "describe"},
				{Type: openai.PartTypeImageURL, ImageURL: &openai.ImageURL{URL: dataURL}},
				{Type: openai.PartTypeImageURL, ImageURL: &openai.ImageURL{URL: "https://x/y.jpg"}},
			}},
		}},
	}
	got, err := TranslateRequest(src, 0)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	want := []anthropic.ContentBlock{
		{Type: anthropic.BlockText, Text: "describe"},
		{Type: anthropic.BlockImage, Source: &anthropic.ImageSource{Type: anthropic.SourceBase64, MediaType: "image/png", Data: "aGVsbG8="}},
		{Type: anthropic.BlockImage, Source: &anthropic.ImageSource{Type: anthropic.SourceURL, URL: "https://x/y.jpg"}},
	}
	if !reflect.DeepEqual(got.Messages[0].Content, want) {
		t.Fatalf("image blocks mismatch:\n got %#v\nwant %#v", got.Messages[0].Content, want)
	}
}

func TestTranslateRequestTools(t *testing.T) {
	params := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)
	src := &openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
		Tools: []openai.Tool{
			{Type: openai.ToolTypeFunction, Function: openai.FunctionSchema{Name: "get_weather", Description: "weather", Parameters: params}},
			{Type: openai.ToolTypeFunction, Function: openai.FunctionSchema{Name: "noparams"}},
		},
	}
	got, err := TranslateRequest(src, 0)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	want := []anthropic.Tool{
		{Name: "get_weather", Description: "weather", InputSchema: params},
		{Name: "noparams", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}
	if !reflect.DeepEqual(got.Tools, want) {
		t.Fatalf("tools mismatch:\n got %#v\nwant %#v", got.Tools, want)
	}
}

func TestTranslateRequestToolChoice(t *testing.T) {
	cases := []struct {
		name string
		in   *openai.ToolChoice
		want *anthropic.ToolChoice
	}{
		{"nil", nil, nil},
		{"auto", &openai.ToolChoice{Mode: openai.ToolChoiceAuto}, &anthropic.ToolChoice{Type: anthropic.ToolChoiceAuto}},
		{"required", &openai.ToolChoice{Mode: openai.ToolChoiceRequired}, &anthropic.ToolChoice{Type: anthropic.ToolChoiceAny}},
		{"none", &openai.ToolChoice{Mode: openai.ToolChoiceNone}, &anthropic.ToolChoice{Type: anthropic.ToolChoiceNone}},
		{"function", &openai.ToolChoice{Function: &openai.ToolChoiceFunction{Name: "f"}}, &anthropic.ToolChoice{Type: anthropic.ToolChoiceTool, Name: "f"}},
		{"unknown", &openai.ToolChoice{Mode: "weird"}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := translateToolChoice(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestResolveMaxTokens(t *testing.T) {
	cases := []struct {
		name     string
		req      *openai.ChatCompletionRequest
		fallback int
		want     int
	}{
		{"completion tokens win", &openai.ChatCompletionRequest{MaxCompletionTokens: intPtr(10), MaxTokens: intPtr(20)}, 0, 10},
		{"max tokens fallback", &openai.ChatCompletionRequest{MaxTokens: intPtr(20)}, 0, 20},
		{"default when absent", &openai.ChatCompletionRequest{}, 0, DefaultMaxTokens},
		{"custom fallback", &openai.ChatCompletionRequest{}, 1000, 1000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveMaxTokens(tc.req, tc.fallback); got != tc.want {
				t.Fatalf("got %d want %d", got, tc.want)
			}
		})
	}
}

func TestClampTemperature(t *testing.T) {
	cases := []struct {
		in   *float64
		want *float64
	}{
		{nil, nil},
		{floatPtr(0.5), floatPtr(0.5)},
		{floatPtr(2.0), floatPtr(1.0)},
		{floatPtr(-1), floatPtr(0)},
	}
	for _, tc := range cases {
		got := clampTemperature(tc.in)
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("clamp(%v) = %v want %v", tc.in, got, tc.want)
		}
	}
}

func TestTranslateRequestPassesSamplingAndStop(t *testing.T) {
	src := &openai.ChatCompletionRequest{
		Model:       "glm-4.6",
		Messages:    []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
		TopP:        floatPtr(0.9),
		Temperature: floatPtr(0.3),
		Stop:        openai.Stop{"STOP"},
		Stream:      true,
	}
	got, err := TranslateRequest(src, 0)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if got.TopP == nil || *got.TopP != 0.9 {
		t.Fatalf("top_p = %v", got.TopP)
	}
	if !reflect.DeepEqual(got.StopSequences, []string{"STOP"}) {
		t.Fatalf("stop = %v", got.StopSequences)
	}
	if !got.Stream {
		t.Fatal("stream not propagated")
	}
}

func TestTranslateRequestUnsupportedRole(t *testing.T) {
	src := &openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: "developer", Content: openai.Content{Text: "x"}}},
	}
	if _, err := TranslateRequest(src, 0); err == nil {
		t.Fatal("expected error for unsupported role")
	}
}
