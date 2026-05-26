package anthropic2openai

import (
	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

// TranslateResponse converts a non-streaming Anthropic Messages response into an
// OpenAI Chat Completions response. created is the Unix-seconds timestamp echoed
// to the client and is supplied by the caller for deterministic output.
func TranslateResponse(src *anthropic.MessagesResponse, created int64) *openai.ChatCompletionResponse {
	text, toolCalls := collectContent(src.Content)
	hasToolCalls := len(toolCalls) > 0
	return &openai.ChatCompletionResponse{
		ID:      completionID(src.ID),
		Object:  openai.ObjectChatCompletion,
		Created: created,
		Model:   src.Model,
		Choices: []openai.Choice{{
			Index: 0,
			Message: openai.ResponseMessage{
				Role:      openai.RoleAssistant,
				Content:   textContent(text, hasToolCalls),
				ToolCalls: toolCalls,
			},
			FinishReason: mapStopReason(src.StopReason, hasToolCalls),
		}},
		Usage: mapUsage(src.Usage),
	}
}

// collectContent splits Anthropic content blocks into the assistant's text and
// its tool calls.
func collectContent(blocks []anthropic.ContentBlock) (string, []openai.ToolCall) {
	var text string
	var toolCalls []openai.ToolCall
	for _, block := range blocks {
		switch block.Type {
		case anthropic.BlockText:
			text += block.Text
		case anthropic.BlockToolUse:
			toolCalls = append(toolCalls, toolCallFromBlock(block))
		}
	}
	return text, toolCalls
}

// toolCallFromBlock converts a tool_use block into an OpenAI tool call.
func toolCallFromBlock(block anthropic.ContentBlock) openai.ToolCall {
	arguments := string(block.Input)
	if arguments == "" {
		arguments = string(emptyJSONObject)
	}
	return openai.ToolCall{
		ID:   block.ID,
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      block.Name,
			Arguments: arguments,
		},
	}
}

// textContent returns a pointer to the text, or nil when the response carries
// only tool calls, matching OpenAI's use of null content in that case.
func textContent(text string, hasToolCalls bool) *string {
	if text == "" && hasToolCalls {
		return nil
	}
	return &text
}

// mapStopReason maps an Anthropic stop reason to an OpenAI finish reason. A
// response containing tool calls always finishes with "tool_calls".
func mapStopReason(reason string, hasToolCalls bool) string {
	if hasToolCalls {
		return openai.FinishReasonToolCalls
	}
	switch reason {
	case anthropic.StopMaxTokens:
		return openai.FinishReasonLength
	case anthropic.StopToolUse:
		return openai.FinishReasonToolCalls
	default:
		return openai.FinishReasonStop
	}
}

// mapUsage converts Anthropic token counts to the OpenAI usage shape.
func mapUsage(u anthropic.Usage) openai.Usage {
	return openai.Usage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.InputTokens + u.OutputTokens,
	}
}
