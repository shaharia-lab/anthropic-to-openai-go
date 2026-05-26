package openai

import (
	"encoding/json"
	"testing"
)

func TestNewErrorResponse(t *testing.T) {
	r := NewErrorResponse("boom", ErrorTypeUpstream)
	if r.Error.Message != "boom" || r.Error.Type != ErrorTypeUpstream {
		t.Fatalf("got %+v", r.Error)
	}
}

func TestStopMarshal(t *testing.T) {
	data, err := json.Marshal(Stop{"a", "b"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(data) != `["a","b"]` {
		t.Fatalf("got %s", data)
	}
}

func TestStopUnmarshalInvalid(t *testing.T) {
	var s Stop
	if err := json.Unmarshal([]byte(`123`), &s); err == nil {
		t.Fatal("expected error for numeric stop value")
	}
}

func TestToolChoiceUnmarshalInvalid(t *testing.T) {
	var tc ToolChoice
	if err := json.Unmarshal([]byte(`{`), &tc); err == nil {
		t.Fatal("expected error for malformed tool_choice")
	}
}

func TestToolChoiceUnmarshalNull(t *testing.T) {
	var tc ToolChoice
	if err := json.Unmarshal([]byte(`null`), &tc); err != nil {
		t.Fatalf("unmarshal null: %v", err)
	}
	if tc.Mode != "" || tc.Function != nil {
		t.Fatalf("expected zero tool choice, got %+v", tc)
	}
}
