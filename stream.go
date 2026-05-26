package anthropic2openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/shaharia-lab/anthropic-to-openai-go/anthropic"
	"github.com/shaharia-lab/anthropic-to-openai-go/openai"
)

// SSE and scanner constants.
const (
	streamDone           = "[DONE]"
	sseDataPrefix        = "data:"
	scannerInitialBuffer = 64 * 1024
	scannerMaxBuffer     = 10 * 1024 * 1024
)

// flusher is implemented by writers that can flush buffered data, allowing
// streamed chunks to reach clients promptly.
type flusher interface {
	Flush()
}

// StreamParams configures translation of a streamed response.
type StreamParams struct {
	// Model is echoed back in every chunk.
	Model string
	// ID is the completion identifier; one is generated when empty.
	ID string
	// Created is the Unix-seconds timestamp echoed in every chunk.
	Created int64
	// IncludeUsage appends a final usage-only chunk, mirroring OpenAI's
	// stream_options.include_usage.
	IncludeUsage bool
}

// TranslateStream reads an Anthropic Messages SSE stream from src and writes an
// OpenAI Chat Completions SSE stream to dst. When dst implements Flush it is
// flushed after every chunk. It returns the first unrecoverable error.
func TranslateStream(dst io.Writer, src io.Reader, params StreamParams) error {
	t := newStreamTranslator(dst, params)
	scanner := bufio.NewScanner(src)
	scanner.Buffer(make([]byte, 0, scannerInitialBuffer), scannerMaxBuffer)
	for scanner.Scan() {
		payload, ok := sseData(scanner.Text())
		if !ok {
			continue
		}
		stop, err := t.consume([]byte(payload))
		if err != nil {
			return err
		}
		if stop {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("anthropic2openai: reading stream: %w", err)
	}
	return t.finishStream()
}

// sseData extracts the payload of a `data:` SSE line, returning ok=false for
// other lines (event:, id:, comments, blanks) and stream terminators.
func sseData(line string) (string, bool) {
	if !strings.HasPrefix(line, sseDataPrefix) {
		return "", false
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, sseDataPrefix))
	if payload == "" || payload == streamDone {
		return "", false
	}
	return payload, true
}

// streamTranslator holds the state required to convert an Anthropic event
// stream into OpenAI chunks.
type streamTranslator struct {
	dst         io.Writer
	flush       func()
	params      StreamParams
	toolIndexes map[int]int // Anthropic block index -> OpenAI tool-call index
	nextTool    int
	inputTokens int
	outTokens   int
	roleSent    bool
	finishSent  bool
	doneSent    bool
	finish      string
}

// newStreamTranslator initialises a translator, generating a completion ID when
// none was supplied and wiring up flushing when supported by dst.
func newStreamTranslator(dst io.Writer, params StreamParams) *streamTranslator {
	if params.ID == "" {
		params.ID = completionID("")
	}
	t := &streamTranslator{
		dst:         dst,
		params:      params,
		toolIndexes: make(map[int]int),
	}
	if f, ok := dst.(flusher); ok {
		t.flush = f.Flush
	}
	return t
}

// consume processes one decoded event, returning stop=true when the stream is
// complete.
func (t *streamTranslator) consume(data []byte) (stop bool, err error) {
	var event anthropic.StreamEvent
	if err = json.Unmarshal(data, &event); err != nil {
		return false, fmt.Errorf("anthropic2openai: decoding stream event: %w", err)
	}
	switch event.Type {
	case anthropic.EventMessageStart:
		t.onMessageStart(event)
	case anthropic.EventContentBlockStart:
		err = t.onContentBlockStart(event)
	case anthropic.EventContentBlockDelta:
		err = t.onContentBlockDelta(event)
	case anthropic.EventMessageDelta:
		err = t.onMessageDelta(event)
	case anthropic.EventMessageStop:
		return true, nil
	case anthropic.EventError:
		return false, streamEventError(event)
	}
	return false, err
}

// onMessageStart records the prompt token count for later usage reporting.
func (t *streamTranslator) onMessageStart(event anthropic.StreamEvent) {
	if event.Message != nil {
		t.inputTokens = event.Message.Usage.InputTokens
	}
}

// onContentBlockStart emits the tool-call header (id and name) when a tool_use
// block begins. Text blocks need no start chunk.
func (t *streamTranslator) onContentBlockStart(event anthropic.StreamEvent) error {
	block := event.ContentBlock
	if block == nil || block.Type != anthropic.BlockToolUse {
		return nil
	}
	if err := t.ensureRole(); err != nil {
		return err
	}
	oaIndex := t.nextTool
	t.nextTool++
	t.toolIndexes[event.Index] = oaIndex
	delta := openai.Delta{ToolCalls: []openai.ToolCallDelta{{
		Index:    oaIndex,
		ID:       block.ID,
		Type:     openai.ToolTypeFunction,
		Function: openai.FunctionCallDelta{Name: block.Name},
	}}}
	return t.writeChunk(t.newChunk(delta, nil))
}

// onContentBlockDelta emits text or tool-argument deltas.
func (t *streamTranslator) onContentBlockDelta(event anthropic.StreamEvent) error {
	if event.Delta == nil {
		return nil
	}
	switch event.Delta.Type {
	case anthropic.DeltaText:
		return t.emitText(event.Delta.Text)
	case anthropic.DeltaInputJSON:
		return t.emitToolArguments(event.Index, event.Delta.PartialJSON)
	default:
		return nil
	}
}

// emitText streams a text delta.
func (t *streamTranslator) emitText(text string) error {
	if text == "" {
		return nil
	}
	if err := t.ensureRole(); err != nil {
		return err
	}
	return t.writeChunk(t.newChunk(openai.Delta{Content: text}, nil))
}

// emitToolArguments streams an incremental tool-argument fragment.
func (t *streamTranslator) emitToolArguments(blockIndex int, partial string) error {
	if partial == "" {
		return nil
	}
	oaIndex, ok := t.toolIndexes[blockIndex]
	if !ok {
		return nil
	}
	delta := openai.Delta{ToolCalls: []openai.ToolCallDelta{{
		Index:    oaIndex,
		Function: openai.FunctionCallDelta{Arguments: partial},
	}}}
	return t.writeChunk(t.newChunk(delta, nil))
}

// onMessageDelta records the output token count and final stop reason, then
// emits the finish chunk.
func (t *streamTranslator) onMessageDelta(event anthropic.StreamEvent) error {
	if event.Usage != nil {
		t.outTokens = event.Usage.OutputTokens
	}
	if event.Delta != nil && event.Delta.StopReason != "" {
		t.finish = mapStopReason(event.Delta.StopReason, t.nextTool > 0)
	}
	return t.emitFinish()
}

// ensureRole emits the initial role delta exactly once.
func (t *streamTranslator) ensureRole() error {
	if t.roleSent {
		return nil
	}
	t.roleSent = true
	return t.writeChunk(t.newChunk(openai.Delta{Role: openai.RoleAssistant}, nil))
}

// emitFinish writes the terminating choice chunk with a finish reason, once.
func (t *streamTranslator) emitFinish() error {
	if t.finishSent {
		return nil
	}
	if err := t.ensureRole(); err != nil {
		return err
	}
	t.finishSent = true
	if t.finish == "" {
		t.finish = openai.FinishReasonStop
	}
	reason := t.finish
	return t.writeChunk(t.newChunk(openai.Delta{}, &reason))
}

// finishStream writes any outstanding finish, usage and terminator chunks.
func (t *streamTranslator) finishStream() error {
	if err := t.emitFinish(); err != nil {
		return err
	}
	if t.params.IncludeUsage {
		if err := t.writeUsageChunk(); err != nil {
			return err
		}
	}
	return t.writeDone()
}

// writeUsageChunk writes a final chunk carrying token usage and no choices.
func (t *streamTranslator) writeUsageChunk() error {
	chunk := openai.ChatCompletionChunk{
		ID:      t.params.ID,
		Object:  openai.ObjectChatCompletionChunk,
		Created: t.params.Created,
		Model:   t.params.Model,
		Choices: []openai.ChunkChoice{},
		Usage: &openai.Usage{
			PromptTokens:     t.inputTokens,
			CompletionTokens: t.outTokens,
			TotalTokens:      t.inputTokens + t.outTokens,
		},
	}
	return t.writeChunk(chunk)
}

// newChunk builds a single-choice chunk with the shared envelope fields.
func (t *streamTranslator) newChunk(delta openai.Delta, finishReason *string) openai.ChatCompletionChunk {
	return openai.ChatCompletionChunk{
		ID:      t.params.ID,
		Object:  openai.ObjectChatCompletionChunk,
		Created: t.params.Created,
		Model:   t.params.Model,
		Choices: []openai.ChunkChoice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finishReason,
		}},
	}
}

// writeChunk encodes and writes a chunk as an SSE data event.
func (t *streamTranslator) writeChunk(chunk openai.ChatCompletionChunk) error {
	payload, err := json.Marshal(chunk)
	if err != nil {
		return fmt.Errorf("anthropic2openai: encoding chunk: %w", err)
	}
	return t.writeSSE(payload)
}

// writeSSE writes a single SSE data event and flushes when supported.
func (t *streamTranslator) writeSSE(payload []byte) error {
	if _, err := fmt.Fprintf(t.dst, "data: %s\n\n", payload); err != nil {
		return fmt.Errorf("anthropic2openai: writing stream: %w", err)
	}
	if t.flush != nil {
		t.flush()
	}
	return nil
}

// writeDone writes the OpenAI stream terminator, once.
func (t *streamTranslator) writeDone() error {
	if t.doneSent {
		return nil
	}
	t.doneSent = true
	return t.writeSSE([]byte(streamDone))
}

// streamEventError converts an upstream error event into a Go error.
func streamEventError(event anthropic.StreamEvent) error {
	if event.Error == nil {
		return fmt.Errorf("anthropic2openai: upstream stream error")
	}
	return fmt.Errorf("anthropic2openai: upstream stream error: %s: %s", event.Error.Type, event.Error.Message)
}
