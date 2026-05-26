package anthropic

// Server-sent event type names emitted while streaming.
const (
	EventMessageStart      = "message_start"
	EventContentBlockStart = "content_block_start"
	EventContentBlockDelta = "content_block_delta"
	EventContentBlockStop  = "content_block_stop"
	EventMessageDelta      = "message_delta"
	EventMessageStop       = "message_stop"
	EventPing              = "ping"
	EventError             = "error"
)

// Delta value types carried by content_block_delta events.
const (
	DeltaText      = "text_delta"
	DeltaInputJSON = "input_json_delta"
)

// StreamEvent is a decoded Anthropic streaming event. Only the fields relevant
// to Type are populated.
type StreamEvent struct {
	Type         string            `json:"type"`
	Index        int               `json:"index"`
	Message      *MessagesResponse `json:"message,omitempty"`
	ContentBlock *ContentBlock     `json:"content_block,omitempty"`
	Delta        *EventDelta       `json:"delta,omitempty"`
	Usage        *Usage            `json:"usage,omitempty"`
	Error        *ErrorBody        `json:"error,omitempty"`
}

// EventDelta carries the incremental payload of streaming events. Text and
// PartialJSON appear in content_block_delta events; StopReason and StopSequence
// appear in message_delta events.
type EventDelta struct {
	Type         string  `json:"type,omitempty"`
	Text         string  `json:"text,omitempty"`
	PartialJSON  string  `json:"partial_json,omitempty"`
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}
