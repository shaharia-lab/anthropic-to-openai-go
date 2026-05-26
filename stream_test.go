package anthropic2openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

// collectChunks parses an OpenAI SSE stream into decoded chunks, asserting it
// ends with the [DONE] terminator.
func collectChunks(t *testing.T, raw string) []openai.ChatCompletionChunk {
	t.Helper()
	var chunks []openai.ChatCompletionChunk
	sawDone := false
	for _, line := range strings.Split(raw, "\n") {
		if !strings.HasPrefix(line, sseDataPrefix) {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, sseDataPrefix))
		if payload == streamDone {
			sawDone = true
			continue
		}
		var chunk openai.ChatCompletionChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("decode chunk %q: %v", payload, err)
		}
		chunks = append(chunks, chunk)
	}
	if !sawDone {
		t.Fatal("stream did not end with [DONE]")
	}
	return chunks
}

const textStreamFixture = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"glm-4.6","usage":{"input_tokens":9,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`

func TestTranslateStreamText(t *testing.T) {
	var out strings.Builder
	err := TranslateStream(&out, strings.NewReader(textStreamFixture), StreamParams{Model: "glm-4.6", ID: "chatcmpl-x", Created: 100})
	if err != nil {
		t.Fatalf("translate stream: %v", err)
	}
	chunks := collectChunks(t, out.String())

	if chunks[0].Choices[0].Delta.Role != openai.RoleAssistant {
		t.Fatalf("first chunk should carry role, got %+v", chunks[0].Choices[0].Delta)
	}
	var text strings.Builder
	var finish string
	for _, c := range chunks {
		if len(c.Choices) == 0 {
			continue
		}
		text.WriteString(c.Choices[0].Delta.Content)
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
		if c.ID != "chatcmpl-x" || c.Object != openai.ObjectChatCompletionChunk {
			t.Fatalf("envelope mismatch: %+v", c)
		}
	}
	if text.String() != "Hello world" {
		t.Fatalf("text = %q", text.String())
	}
	if finish != openai.FinishReasonStop {
		t.Fatalf("finish = %q", finish)
	}
}

func TestTranslateStreamIncludesUsage(t *testing.T) {
	var out strings.Builder
	err := TranslateStream(&out, strings.NewReader(textStreamFixture), StreamParams{Model: "glm-4.6", IncludeUsage: true})
	if err != nil {
		t.Fatalf("translate stream: %v", err)
	}
	chunks := collectChunks(t, out.String())
	last := chunks[len(chunks)-1]
	if last.Usage == nil {
		t.Fatal("expected final usage chunk")
	}
	if last.Usage.PromptTokens != 9 || last.Usage.CompletionTokens != 5 || last.Usage.TotalTokens != 14 {
		t.Fatalf("usage = %+v", last.Usage)
	}
	if len(last.Choices) != 0 {
		t.Fatalf("usage chunk should have no choices, got %d", len(last.Choices))
	}
}

const toolStreamFixture = `event: message_start
data: {"type":"message_start","message":{"id":"msg_2","model":"glm-4.6","usage":{"input_tokens":20,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"NYC\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}

event: message_stop
data: {"type":"message_stop"}

`

func TestTranslateStreamToolCall(t *testing.T) {
	var out strings.Builder
	err := TranslateStream(&out, strings.NewReader(toolStreamFixture), StreamParams{Model: "glm-4.6"})
	if err != nil {
		t.Fatalf("translate stream: %v", err)
	}
	chunks := collectChunks(t, out.String())

	var name, args, finish string
	for _, c := range chunks {
		if len(c.Choices) == 0 {
			continue
		}
		for _, call := range c.Choices[0].Delta.ToolCalls {
			if call.Function.Name != "" {
				name = call.Function.Name
			}
			args += call.Function.Arguments
			if call.ID != "" && (call.Type != openai.ToolTypeFunction || call.Index != 0) {
				t.Fatalf("unexpected tool header: %+v", call)
			}
		}
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if name != "get_weather" {
		t.Fatalf("name = %q", name)
	}
	if args != `{"city":"NYC"}` {
		t.Fatalf("args = %q", args)
	}
	if finish != openai.FinishReasonToolCalls {
		t.Fatalf("finish = %q", finish)
	}
}

func TestTranslateStreamUpstreamError(t *testing.T) {
	fixture := `event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"slow down"}}

`
	var out strings.Builder
	err := TranslateStream(&out, strings.NewReader(fixture), StreamParams{Model: "glm-4.6"})
	if err == nil {
		t.Fatal("expected error from error event")
	}
	if !strings.Contains(err.Error(), "slow down") {
		t.Fatalf("error = %v", err)
	}
}

func TestTranslateStreamMalformedEvent(t *testing.T) {
	fixture := "data: {not valid json}\n\n"
	var out strings.Builder
	if err := TranslateStream(&out, strings.NewReader(fixture), StreamParams{}); err == nil {
		t.Fatal("expected decode error")
	}
}

func TestTranslateStreamGeneratesIDWhenEmpty(t *testing.T) {
	var out strings.Builder
	if err := TranslateStream(&out, strings.NewReader(textStreamFixture), StreamParams{}); err != nil {
		t.Fatalf("translate stream: %v", err)
	}
	chunks := collectChunks(t, out.String())
	if !strings.HasPrefix(chunks[0].ID, completionIDPrefix) {
		t.Fatalf("generated id = %q", chunks[0].ID)
	}
}

func TestSSEData(t *testing.T) {
	cases := []struct {
		line    string
		payload string
		ok      bool
	}{
		{"data: {}", "{}", true},
		{"data:{}", "{}", true},
		{"event: ping", "", false},
		{": comment", "", false},
		{"data: [DONE]", "", false},
		{"data:   ", "", false},
	}
	for _, tc := range cases {
		payload, ok := sseData(tc.line)
		if ok != tc.ok || payload != tc.payload {
			t.Fatalf("sseData(%q) = (%q,%v) want (%q,%v)", tc.line, payload, ok, tc.payload, tc.ok)
		}
	}
}
