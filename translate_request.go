package anthropic2openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

// DefaultMaxTokens is applied when a request omits a token limit, since the
// Anthropic Messages API requires max_tokens.
const DefaultMaxTokens = 4096

// maxAnthropicTemperature is the upper bound the Messages API accepts; OpenAI
// allows up to 2, so values are clamped to keep upstream requests valid.
const maxAnthropicTemperature = 1.0

// emptyObjectSchema is the JSON Schema used for tools that declare no
// parameters, since Anthropic requires an input schema.
var emptyObjectSchema = json.RawMessage(`{"type":"object"}`)

// emptyJSONObject is the canonical empty JSON object used for tool inputs.
var emptyJSONObject = json.RawMessage(`{}`)

// TranslateRequest converts an OpenAI Chat Completions request into an
// equivalent Anthropic Messages request. defaultMaxTokens supplies max_tokens
// when the source omits it; pass 0 to use DefaultMaxTokens.
func TranslateRequest(src *openai.ChatCompletionRequest, defaultMaxTokens int) (*anthropic.MessagesRequest, error) {
	if src == nil {
		return nil, fmt.Errorf("anthropic2openai: nil request")
	}
	system, messages, err := translateMessages(src.Messages)
	if err != nil {
		return nil, err
	}
	return &anthropic.MessagesRequest{
		Model:         src.Model,
		Messages:      messages,
		System:        system,
		MaxTokens:     resolveMaxTokens(src, defaultMaxTokens),
		Temperature:   clampTemperature(src.Temperature),
		TopP:          src.TopP,
		StopSequences: src.Stop,
		Stream:        src.Stream,
		Tools:         translateTools(src.Tools),
		ToolChoice:    translateToolChoice(src.ToolChoice),
	}, nil
}

// resolveMaxTokens picks the effective token limit, preferring the modern
// max_completion_tokens field, then max_tokens, then the fallback.
func resolveMaxTokens(src *openai.ChatCompletionRequest, fallback int) int {
	if fallback <= 0 {
		fallback = DefaultMaxTokens
	}
	switch {
	case src.MaxCompletionTokens != nil && *src.MaxCompletionTokens > 0:
		return *src.MaxCompletionTokens
	case src.MaxTokens != nil && *src.MaxTokens > 0:
		return *src.MaxTokens
	default:
		return fallback
	}
}

// clampTemperature constrains the temperature to the range Anthropic accepts.
func clampTemperature(t *float64) *float64 {
	if t == nil {
		return nil
	}
	v := *t
	switch {
	case v > maxAnthropicTemperature:
		v = maxAnthropicTemperature
	case v < 0:
		v = 0
	}
	return &v
}

// translateMessages splits OpenAI system messages into the Anthropic system
// field and converts the remaining turns, merging consecutive same-role turns
// as the Messages API expects.
func translateMessages(src []openai.Message) (string, []anthropic.Message, error) {
	var systemParts []string
	builder := &conversationBuilder{}
	for i := range src {
		msg := src[i]
		if msg.Role == openai.RoleSystem {
			systemParts = append(systemParts, contentText(msg.Content))
			continue
		}
		role, blocks, err := translateTurn(msg)
		if err != nil {
			return "", nil, fmt.Errorf("anthropic2openai: message %d: %w", i, err)
		}
		builder.add(role, blocks)
	}
	return strings.Join(systemParts, "\n\n"), builder.messages, nil
}

// translateTurn converts a single non-system message into an Anthropic role and
// content blocks.
func translateTurn(msg openai.Message) (string, []anthropic.ContentBlock, error) {
	switch msg.Role {
	case openai.RoleUser:
		blocks, err := userBlocks(msg.Content)
		return anthropic.RoleUser, blocks, err
	case openai.RoleAssistant:
		blocks, err := assistantBlocks(msg)
		return anthropic.RoleAssistant, blocks, err
	case openai.RoleTool:
		block, err := toolResultBlock(msg)
		if err != nil {
			return "", nil, err
		}
		return anthropic.RoleUser, []anthropic.ContentBlock{block}, nil
	default:
		return "", nil, fmt.Errorf("unsupported role %q", msg.Role)
	}
}

// userBlocks converts user content (string or multimodal parts) into blocks.
func userBlocks(content openai.Content) ([]anthropic.ContentBlock, error) {
	if len(content.Parts) == 0 {
		return []anthropic.ContentBlock{textBlock(content.Text)}, nil
	}
	blocks := make([]anthropic.ContentBlock, 0, len(content.Parts))
	for _, part := range content.Parts {
		block, err := translatePart(part)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

// translatePart converts a single multimodal content part.
func translatePart(part openai.ContentPart) (anthropic.ContentBlock, error) {
	switch part.Type {
	case openai.PartTypeText:
		return textBlock(part.Text), nil
	case openai.PartTypeImageURL:
		if part.ImageURL == nil {
			return anthropic.ContentBlock{}, fmt.Errorf("image_url part missing image_url field")
		}
		return imageBlock(part.ImageURL.URL)
	default:
		return anthropic.ContentBlock{}, fmt.Errorf("unsupported content part type %q", part.Type)
	}
}

// assistantBlocks converts an assistant message, including any tool calls, into
// content blocks. An empty text block is emitted when there is no content so
// the turn remains valid.
func assistantBlocks(msg openai.Message) ([]anthropic.ContentBlock, error) {
	var blocks []anthropic.ContentBlock
	if text := contentText(msg.Content); text != "" {
		blocks = append(blocks, textBlock(text))
	}
	for _, call := range msg.ToolCalls {
		block, err := toolUseBlock(call)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	if len(blocks) == 0 {
		blocks = append(blocks, textBlock(""))
	}
	return blocks, nil
}

// toolUseBlock converts an assistant tool call into a tool_use block.
func toolUseBlock(call openai.ToolCall) (anthropic.ContentBlock, error) {
	input := json.RawMessage(strings.TrimSpace(call.Function.Arguments))
	if len(input) == 0 {
		input = emptyJSONObject
	}
	if !json.Valid(input) {
		return anthropic.ContentBlock{}, fmt.Errorf("tool call %q has invalid JSON arguments", call.ID)
	}
	return anthropic.ContentBlock{
		Type:  anthropic.BlockToolUse,
		ID:    call.ID,
		Name:  call.Function.Name,
		Input: input,
	}, nil
}

// toolResultBlock converts an OpenAI tool message into a tool_result block,
// encoding the textual result as a JSON string.
func toolResultBlock(msg openai.Message) (anthropic.ContentBlock, error) {
	content, err := json.Marshal(contentText(msg.Content))
	if err != nil {
		return anthropic.ContentBlock{}, fmt.Errorf("encoding tool result: %w", err)
	}
	return anthropic.ContentBlock{
		Type:      anthropic.BlockToolResult,
		ToolUseID: msg.ToolCallID,
		Content:   content,
	}, nil
}

// textBlock builds a text content block.
func textBlock(text string) anthropic.ContentBlock {
	return anthropic.ContentBlock{Type: anthropic.BlockText, Text: text}
}

// contentText extracts the concatenated text of message content, ignoring
// non-text parts such as images.
func contentText(c openai.Content) string {
	if len(c.Parts) == 0 {
		return c.Text
	}
	var b strings.Builder
	for _, part := range c.Parts {
		if part.Type == openai.PartTypeText {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}

// translateTools converts OpenAI function tools into Anthropic tools.
func translateTools(tools []openai.Tool) []anthropic.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropic.Tool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, anthropic.Tool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: toolSchema(tool.Function.Parameters),
		})
	}
	return out
}

// toolSchema returns a valid input schema, defaulting to an empty object schema
// when the source omits or malforms the parameters.
func toolSchema(params json.RawMessage) json.RawMessage {
	if len(params) == 0 || !json.Valid(params) {
		return emptyObjectSchema
	}
	return params
}

// translateToolChoice maps the OpenAI tool-choice field to its Anthropic
// equivalent.
func translateToolChoice(choice *openai.ToolChoice) *anthropic.ToolChoice {
	if choice == nil {
		return nil
	}
	if choice.Function != nil {
		return &anthropic.ToolChoice{Type: anthropic.ToolChoiceTool, Name: choice.Function.Name}
	}
	switch choice.Mode {
	case openai.ToolChoiceRequired:
		return &anthropic.ToolChoice{Type: anthropic.ToolChoiceAny}
	case openai.ToolChoiceNone:
		return &anthropic.ToolChoice{Type: anthropic.ToolChoiceNone}
	case openai.ToolChoiceAuto:
		return &anthropic.ToolChoice{Type: anthropic.ToolChoiceAuto}
	default:
		return nil
	}
}

// conversationBuilder accumulates Anthropic messages, merging the content of
// consecutive turns that share a role.
type conversationBuilder struct {
	messages []anthropic.Message
}

// add appends content for a role, merging into the previous message when the
// role matches.
func (b *conversationBuilder) add(role string, blocks []anthropic.ContentBlock) {
	if n := len(b.messages); n > 0 && b.messages[n-1].Role == role {
		b.messages[n-1].Content = append(b.messages[n-1].Content, blocks...)
		return
	}
	b.messages = append(b.messages, anthropic.Message{Role: role, Content: blocks})
}
