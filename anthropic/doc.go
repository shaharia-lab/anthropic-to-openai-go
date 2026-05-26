// Package anthropic defines the subset of the Anthropic Messages API types
// required to call an Anthropic-compatible endpoint and decode its responses,
// including the server-sent event types emitted while streaming.
//
// The types target the public wire format shared by Anthropic and compatible
// providers such as z.ai. Content blocks use a single flattened struct whose
// relevant fields depend on the Type discriminator.
package anthropic
