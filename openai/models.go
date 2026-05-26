package openai

// Object identifiers for the models API.
const (
	ObjectModel = "model"
	ObjectList  = "list"
)

// modelOwner is reported as the owner of every proxied model.
const modelOwner = "anthropic-to-openai"

// Model describes a single available model.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the response body of GET /v1/models.
type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// NewModelList builds a model list from the given model IDs, stamping each with
// the supplied created timestamp (Unix seconds).
func NewModelList(ids []string, created int64) ModelList {
	data := make([]Model, 0, len(ids))
	for _, id := range ids {
		data = append(data, Model{ID: id, Object: ObjectModel, Created: created, OwnedBy: modelOwner})
	}
	return ModelList{Object: ObjectList, Data: data}
}
