// Package openai defines the subset of the OpenAI Chat Completions API types
// required to accept requests from, and return responses to, OpenAI-compatible
// clients.
//
// The types model the public wire format only; they intentionally avoid
// provider-specific extensions. Polymorphic fields that OpenAI encodes as
// either a scalar or an object (message content, stop sequences, tool choice)
// have custom (un)marshalling so callers can work with a single Go shape.
package openai
