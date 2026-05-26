package openai

import "testing"

func TestNewModelList(t *testing.T) {
	list := NewModelList([]string{"glm-4.6", "glm-4.5v"}, 1700)
	if list.Object != ObjectList {
		t.Fatalf("object = %q", list.Object)
	}
	if len(list.Data) != 2 {
		t.Fatalf("data len = %d", len(list.Data))
	}
	first := list.Data[0]
	if first.ID != "glm-4.6" || first.Object != ObjectModel || first.Created != 1700 || first.OwnedBy != modelOwner {
		t.Fatalf("first model = %+v", first)
	}
}

func TestNewModelListEmpty(t *testing.T) {
	list := NewModelList(nil, 0)
	if list.Object != ObjectList || len(list.Data) != 0 {
		t.Fatalf("expected empty list, got %+v", list)
	}
}
