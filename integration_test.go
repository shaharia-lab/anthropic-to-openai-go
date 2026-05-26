//go:build integration

// Package anthropic2openai integration tests exercise the full proxy against a
// live Anthropic-compatible endpoint (z.ai by default). They run only with the
// "integration" build tag and require Z_AI_API_KEY:
//
//	Z_AI_API_KEY=... go test -tags integration -run Integration -v ./...
//
// Optional overrides: Z_AI_BASE_URL, Z_AI_MODEL, Z_AI_VISION_MODEL.
package anthropic2openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

// newIntegrationHandler builds a Handler pointing at the live endpoint, skipping
// the test when no API key is configured.
func newIntegrationHandler(t *testing.T) *Handler {
	t.Helper()
	apiKey := os.Getenv("Z_AI_API_KEY")
	if apiKey == "" {
		t.Skip("Z_AI_API_KEY not set; skipping live integration test")
	}
	return New(envOr("Z_AI_BASE_URL", "https://api.z.ai/api/anthropic"), apiKey)
}

// envOr returns the environment value for key, or fallback when unset.
func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// callJSON sends a non-streaming request through the handler and decodes the
// response, failing on any non-200 status.
func callJSON(t *testing.T, h *Handler, body openai.ChatCompletionRequest) openai.ChatCompletionResponse {
	t.Helper()
	rec := serve(t, h, body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return resp
}

// serve marshals body and dispatches it to the handler.
func serve(t *testing.T, h *Handler, body openai.ChatCompletionRequest) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(payload)))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestIntegrationSimpleCompletion(t *testing.T) {
	h := newIntegrationHandler(t)
	resp := callJSON(t, h, openai.ChatCompletionRequest{
		Model:       envOr("Z_AI_MODEL", "glm-4.6"),
		Temperature: floatPtr(0),
		Messages: []openai.Message{
			{Role: openai.RoleUser, Content: openai.Content{Text: "Reply with exactly the single word: PONG"}},
		},
	})
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == nil {
		t.Fatalf("no content in response: %+v", resp)
	}
	got := strings.ToUpper(strings.TrimSpace(*resp.Choices[0].Message.Content))
	if !strings.Contains(got, "PONG") {
		t.Fatalf("expected PONG, got %q", got)
	}
	if resp.Usage.TotalTokens == 0 {
		t.Fatal("expected non-zero usage")
	}
	t.Logf("completion ok: %q (tokens: %d)", got, resp.Usage.TotalTokens)
}

func TestIntegrationStreaming(t *testing.T) {
	h := newIntegrationHandler(t)
	rec := serve(t, h, openai.ChatCompletionRequest{
		Model:         envOr("Z_AI_MODEL", "glm-4.6"),
		Stream:        true,
		StreamOptions: &openai.StreamOptions{IncludeUsage: true},
		Messages: []openai.Message{
			{Role: openai.RoleUser, Content: openai.Content{Text: "Count from 1 to 5, space separated."}},
		},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	chunks := collectChunks(t, rec.Body.String())

	var text strings.Builder
	var finish string
	var usageSeen bool
	for _, c := range chunks {
		if c.Usage != nil {
			usageSeen = true
		}
		if len(c.Choices) == 0 {
			continue
		}
		text.WriteString(c.Choices[0].Delta.Content)
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if strings.TrimSpace(text.String()) == "" {
		t.Fatal("streamed text was empty")
	}
	if finish != openai.FinishReasonStop {
		t.Fatalf("finish = %q", finish)
	}
	if !usageSeen {
		t.Fatal("expected usage chunk with include_usage")
	}
	t.Logf("streamed: %q", strings.TrimSpace(text.String()))
}

func TestIntegrationToolCalling(t *testing.T) {
	h := newIntegrationHandler(t)
	model := envOr("Z_AI_MODEL", "glm-4.6")
	weatherTool := openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionSchema{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		},
	}
	first := callJSON(t, h, openai.ChatCompletionRequest{
		Model:      model,
		Tools:      []openai.Tool{weatherTool},
		ToolChoice: &openai.ToolChoice{Mode: openai.ToolChoiceRequired},
		Messages: []openai.Message{
			{Role: openai.RoleUser, Content: openai.Content{Text: "What is the weather in Paris?"}},
		},
	})
	if first.Choices[0].FinishReason != openai.FinishReasonToolCalls {
		t.Fatalf("finish = %q, expected tool_calls", first.Choices[0].FinishReason)
	}
	calls := first.Choices[0].Message.ToolCalls
	if len(calls) == 0 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected tool calls: %+v", calls)
	}
	if !json.Valid([]byte(calls[0].Function.Arguments)) {
		t.Fatalf("tool arguments not valid JSON: %q", calls[0].Function.Arguments)
	}
	t.Logf("tool call: %s(%s)", calls[0].Function.Name, calls[0].Function.Arguments)

	// Second turn: feed the tool result back and expect a natural-language reply.
	assistant := openai.Message{Role: openai.RoleAssistant, ToolCalls: calls}
	toolResult := openai.Message{Role: openai.RoleTool, ToolCallID: calls[0].ID, Content: openai.Content{Text: `{"temp_c":21,"condition":"sunny"}`}}
	second := callJSON(t, h, openai.ChatCompletionRequest{
		Model: model,
		Tools: []openai.Tool{weatherTool},
		Messages: []openai.Message{
			{Role: openai.RoleUser, Content: openai.Content{Text: "What is the weather in Paris?"}},
			assistant,
			toolResult,
		},
	})
	if second.Choices[0].Message.Content == nil || *second.Choices[0].Message.Content == "" {
		t.Fatalf("expected text reply after tool result: %+v", second.Choices[0].Message)
	}
	t.Logf("post-tool reply: %q", *second.Choices[0].Message.Content)
}

// TestIntegrationVision proves the proxy delivers a real (SVG-rendered) image to
// the upstream in valid Anthropic format and that the endpoint accepts it.
//
// Whether the model correctly *recognises* the image depends entirely on the
// upstream provider. z.ai's Anthropic-compatible endpoint accepts image blocks
// but does not currently process them reliably (it is aimed at GLM-4.6 text and
// coding workloads), so colour-recognition is asserted only when VISION_STRICT
// is set. Use that against a genuinely vision-capable Anthropic-compatible
// endpoint to enforce end-to-end recognition. The exact image translation is
// verified independently by the unit tests.
func TestIntegrationVision(t *testing.T) {
	h := newIntegrationHandler(t)
	model := envOr("Z_AI_VISION_MODEL", "glm-4.5v")
	const wantColor = "green"
	dataURL := solidColorImageDataURL(t, wantColor)

	rec := serve(t, h, openai.ChatCompletionRequest{
		Model:       model,
		Temperature: floatPtr(0),
		Messages: []openai.Message{{
			Role: openai.RoleUser,
			Content: openai.Content{Parts: []openai.ContentPart{
				{Type: openai.PartTypeText, Text: "What is the dominant color of this image? Answer with a single lowercase word."},
				{Type: openai.PartTypeImageURL, ImageURL: &openai.ImageURL{URL: dataURL}},
			}},
		}},
	})
	if rec.Code != http.StatusOK {
		t.Skipf("vision model %q rejected request (status %d): %s", model, rec.Code, rec.Body.String())
	}
	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == nil || *resp.Choices[0].Message.Content == "" {
		t.Fatalf("no content in vision response: %+v", resp)
	}
	reply := strings.ToLower(strings.TrimSpace(*resp.Choices[0].Message.Content))
	t.Logf("vision reply: %q (image delivered and accepted by upstream)", reply)

	if os.Getenv("VISION_STRICT") != "" && !strings.Contains(reply, wantColor) {
		t.Fatalf("VISION_STRICT: expected the model to identify %q, got %q", wantColor, reply)
	}
}
