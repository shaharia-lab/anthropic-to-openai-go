package anthropic

import "encoding/json"

// Message roles for the Anthropic Messages API.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Content block type discriminators.
const (
	BlockText       = "text"
	BlockImage      = "image"
	BlockToolUse    = "tool_use"
	BlockToolResult = "tool_result"
)

// Image source type discriminators.
const (
	SourceBase64 = "base64"
	SourceURL    = "url"
)

// Tool-choice type discriminators.
const (
	ToolChoiceAuto = "auto"
	ToolChoiceAny  = "any"
	ToolChoiceTool = "tool"
	ToolChoiceNone = "none"
)

// MessagesRequest is an Anthropic Messages API request body.
type MessagesRequest struct {
	Model         string      `json:"model"`
	Messages      []Message   `json:"messages"`
	MaxTokens     int         `json:"max_tokens"`
	System        string      `json:"system,omitempty"`
	Temperature   *float64    `json:"temperature,omitempty"`
	TopP          *float64    `json:"top_p,omitempty"`
	StopSequences []string    `json:"stop_sequences,omitempty"`
	Stream        bool        `json:"stream,omitempty"`
	Tools         []Tool      `json:"tools,omitempty"`
	ToolChoice    *ToolChoice `json:"tool_choice,omitempty"`
}

// Message is a single conversation turn.
type Message struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock is a typed piece of message content. Only the fields relevant to
// Type are populated; the rest are omitted when encoding.
type ContentBlock struct {
	Type string `json:"type"`

	// Text block.
	Text string `json:"text,omitempty"`

	// Image block.
	Source *ImageSource `json:"source,omitempty"`

	// Tool-use block.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// Tool-result block.
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

// ImageSource locates image data inline (base64) or by remote URL.
type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// Tool declares a callable tool and its JSON Schema input.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// ToolChoice constrains how the model may use tools.
type ToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}
