// Package anthropic2openai exposes any Anthropic-compatible Messages endpoint
// (such as z.ai's GLM models) through an OpenAI-compatible Chat Completions
// interface.
//
// It offers two layers:
//
//   - Pure translation functions — TranslateRequest, TranslateResponse and
//     TranslateStream — that convert between the two wire formats with no I/O,
//     making them simple to unit test and reuse.
//   - An http.Handler, created with New, that decodes an OpenAI request,
//     forwards it to the configured upstream, and re-encodes the result.
//
// The package supports multi-turn conversations, tool (function) calling,
// multimodal image input and streaming. It depends only on the standard
// library.
package anthropic2openai
