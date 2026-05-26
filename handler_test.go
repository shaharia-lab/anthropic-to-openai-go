package anthropic2openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

// newRequest builds an OpenAI chat-completion POST request from a body struct.
func newRequest(t *testing.T, body openai.ChatCompletionRequest) *http.Request {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	return httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(payload)))
}

func TestHandlerNonStreaming(t *testing.T) {
	var gotPath, gotKey, gotVersion string
	var gotUpstream anthropic.MessagesRequest

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get(headerAPIKey)
		gotVersion = r.Header.Get(headerAnthropicVersion)
		_ = json.NewDecoder(r.Body).Decode(&gotUpstream)
		writeJSON(w, http.StatusOK, anthropic.MessagesResponse{
			ID:         "msg_1",
			Model:      "glm-4.6",
			Content:    []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: "hi there"}},
			StopReason: anthropic.StopEndTurn,
			Usage:      anthropic.Usage{InputTokens: 5, OutputTokens: 2},
		})
	}))
	defer upstream.Close()

	h := New(upstream.URL, "secret-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hello"}}},
	}))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if gotPath != messagesPath || gotKey != "secret-key" || gotVersion != defaultAnthropicVersion {
		t.Fatalf("upstream call: path=%q key=%q version=%q", gotPath, gotKey, gotVersion)
	}
	if gotUpstream.MaxTokens != DefaultMaxTokens {
		t.Fatalf("upstream max_tokens = %d", gotUpstream.MaxTokens)
	}

	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Model != "gpt-4o" {
		t.Fatalf("response should echo requested model, got %q", resp.Model)
	}
	if resp.Choices[0].Message.Content == nil || *resp.Choices[0].Message.Content != "hi there" {
		t.Fatalf("content = %v", resp.Choices[0].Message.Content)
	}
}

func TestHandlerStreaming(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req anthropic.MessagesRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if !req.Stream {
			t.Errorf("expected stream=true upstream")
		}
		w.Header().Set(headerContentType, contentTypeEventStream)
		_, _ = io.WriteString(w, textStreamFixture)
	}))
	defer upstream.Close()

	h := New(upstream.URL, "k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hello"}}},
		Stream:   true,
	}))

	if ct := rec.Header().Get(headerContentType); ct != contentTypeEventStream {
		t.Fatalf("content-type = %q", ct)
	}
	chunks := collectChunks(t, rec.Body.String())
	var text strings.Builder
	for _, c := range chunks {
		if len(c.Choices) > 0 {
			text.WriteString(c.Choices[0].Delta.Content)
		}
	}
	if text.String() != "Hello world" {
		t.Fatalf("streamed text = %q", text.String())
	}
}

func TestHandlerUpstreamErrorRelay(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusTooManyRequests, anthropic.ErrorResponse{
			Type:  "error",
			Error: anthropic.ErrorBody{Type: "rate_limit_error", Message: "too many requests"},
		})
	}))
	defer upstream.Close()

	h := New(upstream.URL, "k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	}))

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d", rec.Code)
	}
	var errResp openai.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Message != "too many requests" || errResp.Error.Type != "rate_limit_error" {
		t.Fatalf("error = %+v", errResp.Error)
	}
}

func TestHandlerMalformedUpstreamResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("not json at all"))
	}))
	defer upstream.Close()

	h := New(upstream.URL, "k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	}))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestHandlerPlainTextUpstreamError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "service unavailable", http.StatusServiceUnavailable)
	}))
	defer upstream.Close()

	h := New(upstream.URL, "k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	}))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d", rec.Code)
	}
	var errResp openai.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if !strings.Contains(errResp.Error.Message, "service unavailable") {
		t.Fatalf("message = %q", errResp.Error.Message)
	}
}

func TestHandlerMethodNotAllowed(t *testing.T) {
	h := New("http://upstream.invalid", "k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestHandlerInvalidBody(t *testing.T) {
	h := New("http://upstream.invalid", "k")
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{not json"))
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestHandlerValidationErrors(t *testing.T) {
	cases := []struct {
		name string
		body openai.ChatCompletionRequest
	}{
		{"missing model", openai.ChatCompletionRequest{Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "x"}}}}},
		{"no messages", openai.ChatCompletionRequest{Model: "glm-4.6"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := New("http://upstream.invalid", "k")
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, newRequest(t, tc.body))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d", rec.Code)
			}
		})
	}
}

func TestHandlerMissingAPIKey(t *testing.T) {
	h := New("http://upstream.invalid", "", WithAPIKeyFromRequest(BearerToken))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	}))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestHandlerAPIKeyFromRequest(t *testing.T) {
	var gotKey string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get(headerAPIKey)
		writeJSON(w, http.StatusOK, anthropic.MessagesResponse{Model: "glm-4.6", StopReason: anthropic.StopEndTurn})
	}))
	defer upstream.Close()

	h := New(upstream.URL, "", WithAPIKeyFromRequest(BearerToken))
	req := newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	})
	req.Header.Set(headerAuthorization, bearerPrefix+"client-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if gotKey != "client-key" {
		t.Fatalf("forwarded key = %q", gotKey)
	}
}

func TestHandlerModelMapper(t *testing.T) {
	var gotModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req anthropic.MessagesRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotModel = req.Model
		writeJSON(w, http.StatusOK, anthropic.MessagesResponse{Model: req.Model, StopReason: anthropic.StopEndTurn})
	}))
	defer upstream.Close()

	mapper := func(string) string { return "glm-4.6" }
	h := New(upstream.URL, "k", WithModelMapper(mapper))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	}))

	if gotModel != "glm-4.6" {
		t.Fatalf("upstream model = %q", gotModel)
	}
}

func TestHandlerUpstreamUnreachable(t *testing.T) {
	h := New("http://127.0.0.1:0", "k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, newRequest(t, openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	}))
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestHandlerRequestTooLarge(t *testing.T) {
	h := New("http://upstream.invalid", "k", WithMaxRequestBytes(16))
	big := strings.Repeat("a", 1024)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"`+big+`"}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestBearerToken(t *testing.T) {
	cases := map[string]string{
		"Bearer abc":  "abc",
		"Bearer  abc": "abc",
		"Basic abc":   "",
		"":            "",
	}
	for header, want := range cases {
		r := httptest.NewRequest(http.MethodPost, "/", nil)
		if header != "" {
			r.Header.Set(headerAuthorization, header)
		}
		if got := BearerToken(r); got != want {
			t.Fatalf("BearerToken(%q) = %q want %q", header, got, want)
		}
	}
}
