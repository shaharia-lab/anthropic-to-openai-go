package anthropic2openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

// Default Handler configuration.
const (
	defaultAnthropicVersion = "2023-06-01"
	defaultMaxRequestBytes  = 10 << 20 // 10 MiB
	defaultRequestTimeout   = 5 * time.Minute
	messagesPath            = "/v1/messages"
	errorReadLimit          = 1 << 20 // 1 MiB
)

// HTTP header names and values used when calling the upstream endpoint.
const (
	headerContentType      = "Content-Type"
	headerAnthropicVersion = "anthropic-version"
	headerAPIKey           = "x-api-key"
	headerAuthorization    = "Authorization"
	headerCacheControl     = "Cache-Control"
	contentTypeJSON        = "application/json"
	contentTypeEventStream = "text/event-stream"
	bearerPrefix           = "Bearer "
)

// Handler is an http.Handler that accepts OpenAI Chat Completions requests,
// forwards them to an Anthropic-compatible Messages endpoint, and returns
// OpenAI-shaped responses. The zero value is not usable; construct one with New.
// A Handler is safe for concurrent use by multiple goroutines.
type Handler struct {
	baseURL           string
	apiKey            string
	anthropicVersion  string
	defaultMaxTokens  int
	maxRequestBytes   int64
	httpClient        *http.Client
	modelMapper       func(string) string
	apiKeyFromRequest func(*http.Request) string
	models            []string
}

// ModelsHandler returns an http.Handler that serves GET /v1/models, listing the
// models configured via WithModels in the OpenAI models format. The list is
// advisory: requests are forwarded regardless of whether their model appears.
func (h *Handler) ModelsHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, openai.NewModelList(h.models, nowUnix()))
	})
}

// New creates a Handler that forwards to baseURL (for example,
// "https://api.z.ai/api/anthropic") authenticating with apiKey. apiKey may be
// empty when WithAPIKeyFromRequest is supplied.
func New(baseURL, apiKey string, opts ...Option) *Handler {
	h := &Handler{
		baseURL:          strings.TrimRight(baseURL, "/"),
		apiKey:           apiKey,
		anthropicVersion: defaultAnthropicVersion,
		defaultMaxTokens: DefaultMaxTokens,
		maxRequestBytes:  defaultMaxRequestBytes,
		httpClient:       &http.Client{Timeout: defaultRequestTimeout},
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// ServeHTTP handles a single Chat Completions request.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", openai.ErrorTypeInvalidRequest)
		return
	}
	req, err := h.decodeRequest(w, r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), openai.ErrorTypeInvalidRequest)
		return
	}
	apiKey := h.resolveAPIKey(r)
	if apiKey == "" {
		writeError(w, http.StatusUnauthorized, "missing upstream API key", openai.ErrorTypeAuthentication)
		return
	}
	upstream, err := h.buildUpstreamRequest(r.Context(), req, apiKey)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), openai.ErrorTypeInvalidRequest)
		return
	}
	resp, err := h.httpClient.Do(upstream) //nolint:bodyclose // body is closed by drainAndClose below
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Sprintf("upstream request failed: %v", err), openai.ErrorTypeUpstream)
		return
	}
	defer drainAndClose(resp.Body)

	if resp.StatusCode != http.StatusOK {
		relayUpstreamError(w, resp)
		return
	}
	if req.Stream {
		h.streamResponse(w, resp, req)
		return
	}
	h.jsonResponse(w, resp, req)
}

// decodeRequest reads and validates the OpenAI request, enforcing the body-size
// limit.
func (h *Handler) decodeRequest(w http.ResponseWriter, r *http.Request) (*openai.ChatCompletionRequest, error) {
	body := http.MaxBytesReader(w, r.Body, h.maxRequestBytes)
	var req openai.ChatCompletionRequest
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}
	if req.Model == "" {
		return nil, errors.New("model is required")
	}
	if len(req.Messages) == 0 {
		return nil, errors.New("messages must not be empty")
	}
	return &req, nil
}

// resolveAPIKey returns the upstream key for this request, preferring a
// per-request resolver when configured.
func (h *Handler) resolveAPIKey(r *http.Request) string {
	if h.apiKeyFromRequest != nil {
		return h.apiKeyFromRequest(r)
	}
	return h.apiKey
}

// buildUpstreamRequest translates the request and wraps it in an HTTP request
// carrying the required Anthropic headers. The client's Authorization header is
// never forwarded; only the resolved key is sent as x-api-key.
func (h *Handler) buildUpstreamRequest(ctx context.Context, req *openai.ChatCompletionRequest, apiKey string) (*http.Request, error) {
	translated, err := TranslateRequest(req, h.defaultMaxTokens)
	if err != nil {
		return nil, err
	}
	if h.modelMapper != nil {
		translated.Model = h.modelMapper(translated.Model)
	}
	payload, err := json.Marshal(translated)
	if err != nil {
		return nil, fmt.Errorf("encoding upstream request: %w", err)
	}
	upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, h.baseURL+messagesPath, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("building upstream request: %w", err)
	}
	upstream.Header.Set(headerContentType, contentTypeJSON)
	upstream.Header.Set(headerAnthropicVersion, h.anthropicVersion)
	upstream.Header.Set(headerAPIKey, apiKey)
	return upstream, nil
}

// jsonResponse decodes a non-streaming upstream response and writes the
// translated OpenAI response.
func (h *Handler) jsonResponse(w http.ResponseWriter, resp *http.Response, req *openai.ChatCompletionRequest) {
	var message anthropic.MessagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&message); err != nil {
		writeError(w, http.StatusBadGateway, fmt.Sprintf("decoding upstream response: %v", err), openai.ErrorTypeUpstream)
		return
	}
	out := TranslateResponse(&message, nowUnix())
	out.Model = req.Model
	writeJSON(w, http.StatusOK, out)
}

// streamResponse translates the upstream SSE stream to the client. Because the
// response status is sent before streaming begins, mid-stream failures are
// surfaced as a best-effort SSE error event rather than an HTTP status.
func (h *Handler) streamResponse(w http.ResponseWriter, resp *http.Response, req *openai.ChatCompletionRequest) {
	w.Header().Set(headerContentType, contentTypeEventStream)
	w.Header().Set(headerCacheControl, "no-cache")
	w.WriteHeader(http.StatusOK)

	writer := newFlushWriter(w)
	params := StreamParams{
		Model:        req.Model,
		Created:      nowUnix(),
		IncludeUsage: req.StreamOptions != nil && req.StreamOptions.IncludeUsage,
	}
	if err := TranslateStream(writer, resp.Body, params); err != nil {
		writeStreamError(writer, err)
	}
}

// BearerToken extracts a bearer token from the request's Authorization header.
// It is a convenience for use with WithAPIKeyFromRequest, enabling pass-through
// authentication where each client supplies its own upstream key.
func BearerToken(r *http.Request) string {
	auth := r.Header.Get(headerAuthorization)
	if !strings.HasPrefix(auth, bearerPrefix) {
		return ""
	}
	return strings.TrimSpace(strings.TrimPrefix(auth, bearerPrefix))
}

// relayUpstreamError translates an upstream error body into an OpenAI error,
// preserving the upstream status code.
func relayUpstreamError(w http.ResponseWriter, resp *http.Response) {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, errorReadLimit))
	message := strings.TrimSpace(string(body))
	errType := openai.ErrorTypeUpstream

	var upstreamErr anthropic.ErrorResponse
	if json.Unmarshal(body, &upstreamErr) == nil && upstreamErr.Error.Message != "" {
		message = upstreamErr.Error.Message
		errType = upstreamErr.Error.Type
	}
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	writeError(w, resp.StatusCode, message, errType)
}

// writeStreamError emits an SSE error event followed by the stream terminator.
func writeStreamError(w io.Writer, err error) {
	payload, marshalErr := json.Marshal(openai.NewErrorResponse(err.Error(), openai.ErrorTypeUpstream))
	if marshalErr != nil {
		payload = []byte(`{"error":{"message":"stream error","type":"upstream_error"}}`)
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", payload)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", streamDone)
}

// writeJSON encodes payload as a JSON response with the given status.
func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set(headerContentType, contentTypeJSON)
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

// writeError writes an OpenAI error envelope with the given status.
func writeError(w http.ResponseWriter, status int, message, errType string) {
	writeJSON(w, status, openai.NewErrorResponse(message, errType))
}

// drainAndClose discards a bounded amount of any remaining body and closes it so
// the underlying connection can be reused.
func drainAndClose(body io.ReadCloser) {
	_, _ = io.Copy(io.Discard, io.LimitReader(body, errorReadLimit))
	_ = body.Close()
}

// nowUnix returns the current Unix time in seconds.
func nowUnix() int64 {
	return time.Now().Unix()
}
