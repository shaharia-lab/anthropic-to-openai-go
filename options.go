package anthropic2openai

import "net/http"

// Option configures a Handler. Options are applied in order by New.
type Option func(*Handler)

// WithHTTPClient sets the HTTP client used for upstream requests. A nil client
// is ignored. Provide a client with a timeout suited to your latency budget.
func WithHTTPClient(client *http.Client) Option {
	return func(h *Handler) {
		if client != nil {
			h.httpClient = client
		}
	}
}

// WithAnthropicVersion overrides the anthropic-version request header.
func WithAnthropicVersion(version string) Option {
	return func(h *Handler) {
		if version != "" {
			h.anthropicVersion = version
		}
	}
}

// WithDefaultMaxTokens sets the max_tokens applied when a request omits a token
// limit. Non-positive values are ignored.
func WithDefaultMaxTokens(n int) Option {
	return func(h *Handler) {
		if n > 0 {
			h.defaultMaxTokens = n
		}
	}
}

// WithModelMapper installs a function that rewrites the requested model name
// before it is forwarded upstream (for example, mapping "gpt-4o" to "glm-4.6").
// A nil mapper is ignored.
func WithModelMapper(mapper func(string) string) Option {
	return func(h *Handler) {
		if mapper != nil {
			h.modelMapper = mapper
		}
	}
}

// WithMaxRequestBytes caps the size of accepted request bodies, protecting the
// service from oversized payloads. Non-positive values are ignored.
func WithMaxRequestBytes(n int64) Option {
	return func(h *Handler) {
		if n > 0 {
			h.maxRequestBytes = n
		}
	}
}

// WithModels sets the model IDs advertised by the /v1/models endpoint (served by
// NewServeMux and Handler.ModelsHandler). It does not affect request handling.
func WithModels(ids ...string) Option {
	return func(h *Handler) {
		h.models = append([]string(nil), ids...)
	}
}

// WithAPIKeyFromRequest derives the upstream API key from each incoming request
// instead of using a static key. Returning an empty string rejects the request
// with HTTP 401. BearerToken is a ready-made implementation.
func WithAPIKeyFromRequest(fn func(*http.Request) string) Option {
	return func(h *Handler) {
		h.apiKeyFromRequest = fn
	}
}
