package anthropic2openai

import "net/http"

// Standard OpenAI-compatible route paths.
const (
	PathChatCompletions = "/v1/chat/completions"
	PathModels          = "/v1/models"
)

// NewServeMux returns an *http.ServeMux wired with the standard OpenAI-compatible
// routes (POST /v1/chat/completions and GET /v1/models), forwarding to the given
// Anthropic-compatible endpoint. It is the quickest way to expose a working
// proxy:
//
//	mux := anthropic2openai.NewServeMux("https://api.z.ai/api/anthropic", apiKey,
//		anthropic2openai.WithModels("glm-4.6"))
//	log.Fatal(http.ListenAndServe(":8080", mux))
//
// For finer control — custom paths, middleware, or mounting under an existing
// router — construct a Handler with New and use it directly; Handler implements
// http.Handler.
func NewServeMux(baseURL, apiKey string, opts ...Option) *http.ServeMux {
	h := New(baseURL, apiKey, opts...)
	mux := http.NewServeMux()
	mux.Handle("POST "+PathChatCompletions, h)
	mux.Handle("GET "+PathModels, h.ModelsHandler())
	return mux
}
