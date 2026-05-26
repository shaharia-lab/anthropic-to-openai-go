package openai

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestContentUnmarshalString(t *testing.T) {
	var c Content
	if err := json.Unmarshal([]byte(`"hello"`), &c); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if c.Text != "hello" || len(c.Parts) != 0 {
		t.Fatalf("got text=%q parts=%v", c.Text, c.Parts)
	}
}

func TestContentUnmarshalParts(t *testing.T) {
	data := `[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"http://x/y.png"}}]`
	var c Content
	if err := json.Unmarshal([]byte(data), &c); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if c.Text != "" || len(c.Parts) != 2 {
		t.Fatalf("got text=%q parts=%d", c.Text, len(c.Parts))
	}
	if c.Parts[1].ImageURL == nil || c.Parts[1].ImageURL.URL != "http://x/y.png" {
		t.Fatalf("image part not decoded: %+v", c.Parts[1])
	}
}

func TestContentUnmarshalNull(t *testing.T) {
	var c Content
	if err := json.Unmarshal([]byte(`null`), &c); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !c.IsZero() {
		t.Fatalf("expected zero content, got %+v", c)
	}
}

func TestContentUnmarshalInvalid(t *testing.T) {
	var c Content
	if err := json.Unmarshal([]byte(`42`), &c); err == nil {
		t.Fatal("expected error for numeric content")
	}
}

func TestContentMarshalRoundTrip(t *testing.T) {
	cases := []Content{
		{Text: "plain"},
		{Parts: []ContentPart{{Type: PartTypeText, Text: "p"}}},
	}
	for _, in := range cases {
		data, err := json.Marshal(in)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var out Content
		if err := json.Unmarshal(data, &out); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if !reflect.DeepEqual(in, out) {
			t.Fatalf("round trip mismatch: %+v != %+v", in, out)
		}
	}
}

func TestStopUnmarshal(t *testing.T) {
	cases := map[string]Stop{
		`"x"`:       {"x"},
		`["a","b"]`: {"a", "b"},
		`null`:      nil,
	}
	for input, want := range cases {
		var s Stop
		if err := json.Unmarshal([]byte(input), &s); err != nil {
			t.Fatalf("unmarshal %s: %v", input, err)
		}
		if !reflect.DeepEqual(s, want) {
			t.Fatalf("input %s: got %v want %v", input, s, want)
		}
	}
}

func TestToolChoiceUnmarshalMode(t *testing.T) {
	var tc ToolChoice
	if err := json.Unmarshal([]byte(`"auto"`), &tc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if tc.Mode != "auto" || tc.Function != nil {
		t.Fatalf("got %+v", tc)
	}
}

func TestToolChoiceUnmarshalFunction(t *testing.T) {
	var tc ToolChoice
	data := `{"type":"function","function":{"name":"get_weather"}}`
	if err := json.Unmarshal([]byte(data), &tc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if tc.Mode != "function" || tc.Function == nil || tc.Function.Name != "get_weather" {
		t.Fatalf("got %+v", tc)
	}
}
