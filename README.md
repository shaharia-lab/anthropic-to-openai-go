# anthropic-to-openai

[![Go Reference](https://pkg.go.dev/badge/github.com/shaharia-lab/anthropic-to-openai-go.svg)](https://pkg.go.dev/github.com/shaharia-lab/anthropic-to-openai-go)

Expose any **Anthropic-compatible Messages endpoint** (Anthropic's own API,
[z.ai](https://z.ai)'s GLM models, or any other) through an
**OpenAI-compatible Chat Completions interface**.

Point it at a base URL and an API key, and your existing OpenAI clients and SDKs
can talk to a non-OpenAI model unchanged.

```
OpenAI client ──▶ anthropic-to-openai ──▶ Anthropic-compatible endpoint
  (Chat Completions)   (translates both ways)        (e.g. z.ai GLM)
```

## Features

- **Drop-in `http.Handler`** — mount it in any Go HTTP server.
- **Multi-turn conversations** — system / user / assistant / tool turns, with
  same-role turns merged as the Messages API expects.
- **Tool / function calling** — request tools, assistant `tool_calls`, and
  `tool` result messages, in both directions.
- **Vision / images** — OpenAI `image_url` parts (remote URLs and base64 data
  URLs) become Anthropic image blocks.
- **Streaming** — Server-Sent Events translated to OpenAI
  `chat.completion.chunk` deltas, including streamed tool calls and usage.
- **`/v1/models`** — optional standard models listing.
- **Zero third-party dependencies** — standard library only.
- **Secure & reliable by default** — request body size limits, upstream
  timeouts, context propagation, no secret logging, and the client
  `Authorization` header is never forwarded upstream.

## Install

```sh
go get github.com/shaharia-lab/anthropic-to-openai-go
```

Requires Go 1.23+.

## Quick start

The fastest path — a ready-to-serve mux with the standard OpenAI routes:

```go
package main

import (
	"log"
	"net/http"
	"os"

	a2o "github.com/shaharia-lab/anthropic-to-openai-go"
)

func main() {
	mux := a2o.NewServeMux(
		"https://api.z.ai/api/anthropic", // any Anthropic-compatible base URL
		os.Getenv("Z_AI_API_KEY"),
		a2o.WithModels("glm-4.6"), // advertised by GET /v1/models
	)
	log.Fatal(http.ListenAndServe(":8080", mux))
}
```

Then call it with any OpenAI client:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.6","messages":[{"role":"user","content":"Say hello"}]}'
```

## Mounting the handler yourself

`Handler` implements `http.Handler`, so you can mount it under any router,
add middleware, or choose your own path:

```go
proxy := a2o.New("https://api.z.ai/api/anthropic", apiKey)

mux := http.NewServeMux()
mux.Handle("POST /v1/chat/completions", proxy)
mux.Handle("GET /v1/models", proxy.ModelsHandler())
```

### Pass-through authentication

To let each client supply its own upstream key (instead of a server-wide key),
derive it from the incoming request. `BearerToken` reads the `Authorization`
header:

```go
proxy := a2o.New("https://api.z.ai/api/anthropic", "",
	a2o.WithAPIKeyFromRequest(a2o.BearerToken))
```

## Options

| Option | Purpose |
| --- | --- |
| `WithHTTPClient(*http.Client)` | Custom client / timeouts for upstream calls. |
| `WithAnthropicVersion(string)` | Override the `anthropic-version` header. |
| `WithDefaultMaxTokens(int)` | `max_tokens` used when a request omits one (default 4096). |
| `WithModelMapper(func(string) string)` | Rewrite the model name before forwarding (e.g. `gpt-4o` → `glm-4.6`). |
| `WithMaxRequestBytes(int64)` | Cap accepted request body size (default 10 MiB). |
| `WithModels(...string)` | Models advertised by `/v1/models`. |
| `WithAPIKeyFromRequest(func(*http.Request) string)` | Per-request upstream key resolution. |

## Translation functions

The pure, I/O-free translators are exported for direct use and testing:

```go
anthReq, err := a2o.TranslateRequest(openaiReq, 0)        // OpenAI -> Anthropic
openaiResp   := a2o.TranslateResponse(anthResp, created)  // Anthropic -> OpenAI
err          = a2o.TranslateStream(dst, src, params)      // SSE stream translation
```

## Behaviour notes

- **`max_tokens`** is required by the Messages API. The proxy uses
  `max_completion_tokens`, then `max_tokens`, then the configured default.
- **Temperature** is clamped to the `[0, 1]` range the Messages API accepts
  (OpenAI permits up to 2).
- **Response `model`** echoes the model the client requested.
- **Vision support depends on the upstream provider.** The proxy always
  delivers images in valid Anthropic format, but the upstream model must
  actually support vision. Note that z.ai's *Anthropic-compatible* endpoint
  accepts image blocks yet does not currently process them (it targets GLM-4.6
  text/coding workloads); a genuinely vision-capable Anthropic-compatible
  endpoint works as expected.

## Testing

Unit tests (no network) — they cover the full translation surface and the
handler against a fake upstream:

```sh
go test -race -cover ./...
```

Live integration tests against a real endpoint run only with the `integration`
build tag and an API key:

```sh
Z_AI_API_KEY=... go test -tags integration -run Integration -v ./...
```

Optional environment overrides: `Z_AI_BASE_URL`, `Z_AI_MODEL`,
`Z_AI_VISION_MODEL`, and `VISION_STRICT=1` to assert image recognition against
vision-capable endpoints.

## License

[MIT](LICENSE)
