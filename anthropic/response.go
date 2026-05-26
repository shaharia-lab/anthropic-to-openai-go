package anthropic

// Stop reasons returned by the Messages API.
const (
	StopEndTurn      = "end_turn"
	StopMaxTokens    = "max_tokens"
	StopStopSequence = "stop_sequence"
	StopToolUse      = "tool_use"
)

// MessagesResponse is a non-streaming Messages API response.
type MessagesResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   string         `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        Usage          `json:"usage"`
}

// Usage reports token counts for a response.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ErrorResponse is the Anthropic error envelope.
type ErrorResponse struct {
	Type  string    `json:"type"`
	Error ErrorBody `json:"error"`
}

// ErrorBody describes a single error.
type ErrorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}
