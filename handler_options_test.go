package anthropic2openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

func TestHandlerOptionsApplied(t *testing.T) {
	var gotVersion string
	var gotMaxTokens int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotVersion = r.Header.Get(headerAnthropicVersion)
		var req anthropic.MessagesRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotMaxTokens = req.MaxTokens
		writeJSON(w, http.StatusOK, anthropic.MessagesResponse{Model: "glm-4.6", StopReason: anthropic.StopEndTurn})
	}))
	defer upstream.Close()

	h := New(upstream.URL, "k",
		WithAnthropicVersion("2024-09-01"),
		WithDefaultMaxTokens(99),
		WithHTTPClient(&http.Client{Timeout: 10 * time.Second}),
		WithMaxRequestBytes(1<<20),
	)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	}))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if gotVersion != "2024-09-01" {
		t.Fatalf("version = %q", gotVersion)
	}
	if gotMaxTokens != 99 {
		t.Fatalf("max_tokens = %d", gotMaxTokens)
	}
}

func TestOptionsIgnoreInvalidValues(t *testing.T) {
	// Invalid option values must be ignored, leaving defaults intact.
	h := New("http://upstream.invalid", "k",
		WithHTTPClient(nil),
		WithAnthropicVersion(""),
		WithDefaultMaxTokens(0),
		WithModelMapper(nil),
		WithMaxRequestBytes(0),
	)
	if h.anthropicVersion != defaultAnthropicVersion {
		t.Fatalf("version = %q", h.anthropicVersion)
	}
	if h.defaultMaxTokens != DefaultMaxTokens {
		t.Fatalf("max tokens = %d", h.defaultMaxTokens)
	}
	if h.maxRequestBytes != defaultMaxRequestBytes {
		t.Fatalf("max bytes = %d", h.maxRequestBytes)
	}
	if h.httpClient == nil || h.modelMapper != nil {
		t.Fatal("nil options should not override defaults")
	}
}

func TestHandlerStreamingUpstreamErrorEvent(t *testing.T) {
	fixture := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(headerContentType, contentTypeEventStream)
		_, _ = w.Write([]byte(fixture))
	}))
	defer upstream.Close()

	h := New(upstream.URL, "k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
		Stream:   true,
	}))

	body := rec.Body.String()
	if !strings.Contains(body, "busy") || !strings.Contains(body, streamDone) {
		t.Fatalf("expected error event and terminator, got: %s", body)
	}
}
