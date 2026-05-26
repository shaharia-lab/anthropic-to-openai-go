package anthropic2openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

func TestNewServeMuxRoutesChatCompletions(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, anthropic.MessagesResponse{
			Model:      "glm-4.6",
			Content:    []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: "ok"}},
			StopReason: anthropic.StopEndTurn,
		})
	}))
	defer upstream.Close()

	mux := NewServeMux(upstream.URL, "k")
	body, _ := json.Marshal(openai.ChatCompletionRequest{
		Model:    "glm-4.6",
		Messages: []openai.Message{{Role: openai.RoleUser, Content: openai.Content{Text: "hi"}}},
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathChatCompletions, strings.NewReader(string(body))))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestNewServeMuxModelsEndpoint(t *testing.T) {
	mux := NewServeMux("http://upstream.invalid", "k", WithModels("glm-4.6", "glm-4.5v"))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathModels, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var list openai.ModelList
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if list.Object != openai.ObjectList || len(list.Data) != 2 {
		t.Fatalf("list = %+v", list)
	}
	if list.Data[0].ID != "glm-4.6" || list.Data[0].Object != openai.ObjectModel {
		t.Fatalf("model = %+v", list.Data[0])
	}
}

func TestNewServeMuxRejectsWrongMethod(t *testing.T) {
	mux := NewServeMux("http://upstream.invalid", "k")
	rec := httptest.NewRecorder()
	// GET on the chat-completions path is not routed and must not be 200.
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathChatCompletions, nil))
	if rec.Code == http.StatusOK {
		t.Fatalf("expected non-200 for GET on chat completions, got %d", rec.Code)
	}
}

func TestModelsHandlerEmptyByDefault(t *testing.T) {
	h := New("http://upstream.invalid", "k")
	rec := httptest.NewRecorder()
	h.ModelsHandler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathModels, nil))
	var list openai.ModelList
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if list.Object != openai.ObjectList || len(list.Data) != 0 {
		t.Fatalf("expected empty list, got %+v", list)
	}
}
