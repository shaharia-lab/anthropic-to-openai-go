package openai

import (
	"bytes"
	"encoding/json"
)

// Message roles defined by the OpenAI Chat Completions API.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// ToolTypeFunction is the only tool type currently defined by OpenAI.
const ToolTypeFunction = "function"

// Tool-choice mode strings.
const (
	ToolChoiceNone     = "none"
	ToolChoiceAuto     = "auto"
	ToolChoiceRequired = "required"
)

// ChatCompletionRequest is an OpenAI Chat Completions request body.
type ChatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []Message      `json:"messages"`
	MaxTokens           *int           `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int           `json:"max_completion_tokens,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	Stop                Stop           `json:"stop,omitempty"`
	Stream              bool           `json:"stream,omitempty"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
	Tools               []Tool         `json:"tools,omitempty"`
	ToolChoice          *ToolChoice    `json:"tool_choice,omitempty"`
	User                string         `json:"user,omitempty"`
}

// StreamOptions controls streaming behaviour.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// Message is a single chat message.
type Message struct {
	Role       string     `json:"role"`
	Content    Content    `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is a function call requested by the assistant.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall names a function and carries its JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool declares a function the model may call.
type Tool struct {
	Type     string         `json:"type"`
	Function FunctionSchema `json:"function"`
}

// FunctionSchema describes a callable function and its JSON Schema parameters.
type FunctionSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Stop is the stop-sequence field, encoded by OpenAI as either a single string
// or an array of strings.
type Stop []string

// UnmarshalJSON accepts a string or an array of strings.
func (s *Stop) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, jsonNull) {
		return nil
	}
	if trimmed[0] == '[' {
		var list []string
		if err := json.Unmarshal(trimmed, &list); err != nil {
			return err
		}
		*s = list
		return nil
	}
	var single string
	if err := json.Unmarshal(trimmed, &single); err != nil {
		return err
	}
	*s = []string{single}
	return nil
}

// MarshalJSON always encodes as an array of strings.
func (s Stop) MarshalJSON() ([]byte, error) {
	return json.Marshal([]string(s))
}

// ToolChoice selects how the model uses tools. OpenAI encodes it as either a
// mode string ("none", "auto", "required") or an object naming a function.
type ToolChoice struct {
	Mode     string
	Function *ToolChoiceFunction
}

// ToolChoiceFunction names a function the model is forced to call.
type ToolChoiceFunction struct {
	Name string `json:"name"`
}

// UnmarshalJSON accepts a mode string or a {"type":"function",...} object.
func (t *ToolChoice) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, jsonNull) {
		return nil
	}
	if trimmed[0] == '"' {
		return json.Unmarshal(trimmed, &t.Mode)
	}
	var obj struct {
		Type     string              `json:"type"`
		Function *ToolChoiceFunction `json:"function"`
	}
	if err := json.Unmarshal(trimmed, &obj); err != nil {
		return err
	}
	t.Mode = obj.Type
	t.Function = obj.Function
	return nil
}
