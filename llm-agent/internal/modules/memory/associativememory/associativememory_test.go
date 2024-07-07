package associativememory

import (
	"testing"
	"time"

	"github.com/yourusername/llm-agent/configs"
	"github.com/yourusername/llm-agent/pkg/embeddings"
)

func TestAddEvent(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddEvent(created, &expiration, "subject", "predicate", "object", "description", []string{"keyword1", "keyword2"}, 0.5, "embedding_id", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding event: %v", err)
	}

	// Add more test cases for different scenarios
}

func TestAddThought(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddThought(created, &expiration, "subject", "predicate", "object", "description", []string{"keyword1", "keyword2"}, 0.5, "embedding_id", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding thought: %v", err)
	}

	// Add more test cases for different scenarios
}

func TestAddChat(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddChat(created, &expiration, "subject", "predicate", "object", "description", []string{"keyword1", "keyword2"}, 0.5, "embedding_id", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding chat: %v", err)
	}

	// Add more test cases for different scenarios
}

func TestRetrieveEventsByFilter(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddEvent(created, &expiration, "subject1", "predicate1", "object1", "description1", []string{"keyword1", "keyword2"}, 0.5, "embedding_id1", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding event: %v", err)
	}

	err = am.AddEvent(created, &expiration, "subject2", "predicate2", "object2", "description2", []string{"keyword3", "keyword4"}, 0.7, "embedding_id2", []string{"filling3", "filling4"})
	if err != nil {
		t.Errorf("Error adding event: %v", err)
	}

	filters := []FilterMemory{
		{Subject: "subject1"},
	}
	events, err := am.RetrieveEventsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving events by filter: %v", err)
	}

	if len(events) != 1 {
		t.Errorf("Expected 1 event, got %d", len(events))
	}

	expected := ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if events[0] != expected {
		t.Errorf("Expected %v, got %v", expected, events[0])
	}

	// Test case for multiple filters
	filters = []FilterMemory{
		{Subject: "subject1", Predicate: "predicate1"},
	}
	events, err = am.RetrieveEventsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving events by filter: %v", err)
	}

	if len(events) != 1 {
		t.Errorf("Expected 1 event, got %d", len(events))
	}

	expected = ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if events[0] != expected {
		t.Errorf("Expected %v, got %v", expected, events[0])
	}

	// Test case for non-existent event
	filters = []FilterMemory{
		{Subject: "subject3"},
	}
	events, err = am.RetrieveEventsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving events by filter: %v", err)
	}

	if len(events) != 0 {
		t.Errorf("Expected 0 events, got %d", len(events))
	}

	// Add more test cases for different scenarios
}

func TestRetrieveThoughtsByFilter(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddThought(created, &expiration, "subject1", "predicate1", "object1", "description1", []string{"keyword1", "keyword2"}, 0.5, "embedding_id1", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding thought: %v", err)
	}

	err = am.AddThought(created, &expiration, "subject2", "predicate2", "object2", "description2", []string{"keyword3", "keyword4"}, 0.7, "embedding_id2", []string{"filling3", "filling4"})
	if err != nil {
		t.Errorf("Error adding thought: %v", err)
	}

	filters := []FilterMemory{
		{Subject: "subject1"},
	}
	thoughts, err := am.RetrieveThoughtsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving thoughts by filter: %v", err)
	}

	if len(thoughts) != 1 {
		t.Errorf("Expected 1 thought, got %d", len(thoughts))
	}

	expected := ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if thoughts[0] != expected {
		t.Errorf("Expected %v, got %v", expected, thoughts[0])
	}

	// Test case for multiple filters
	filters = []FilterMemory{
		{Subject: "subject1", Predicate: "predicate1"},
	}
	thoughts, err = am.RetrieveThoughtsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving thoughts by filter: %v", err)
	}

	if len(thoughts) != 1 {
		t.Errorf("Expected 1 thought, got %d", len(thoughts))
	}

	expected = ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if thoughts[0] != expected {
		t.Errorf("Expected %v, got %v", expected, thoughts[0])
	}

	// Test case for non-existent thought
	filters = []FilterMemory{
		{Subject: "subject3"},
	}
	thoughts, err = am.RetrieveThoughtsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving thoughts by filter: %v", err)
	}

	if len(thoughts) != 0 {
		t.Errorf("Expected 0 thoughts, got %d", len(thoughts))
	}

	// Add more test cases for different scenarios
}

func TestRetrieveChatsByFilter(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddChat(created, &expiration, "subject1", "predicate1", "object1", "description1", []string{"keyword1", "keyword2"}, 0.5, "embedding_id1", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding chat: %v", err)
	}

	err = am.AddChat(created, &expiration, "subject2", "predicate2", "object2", "description2", []string{"keyword3", "keyword4"}, 0.7, "embedding_id2", []string{"filling3", "filling4"})
	if err != nil {
		t.Errorf("Error adding chat: %v", err)
	}

	filters := []FilterMemory{
		{Subject: "subject1"},
	}
	chats, err := am.RetrieveChatsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving chats by filter: %v", err)
	}

	if len(chats) != 1 {
		t.Errorf("Expected 1 chat, got %d", len(chats))
	}

	expected := ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if chats[0] != expected {
		t.Errorf("Expected %v, got %v", expected, chats[0])
	}

	// Test case for multiple filters
	filters = []FilterMemory{
		{Subject: "subject1", Predicate: "predicate1"},
	}
	chats, err = am.RetrieveChatsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving chats by filter: %v", err)
	}

	if len(chats) != 1 {
		t.Errorf("Expected 1 chat, got %d", len(chats))
	}

	expected = ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if chats[0] != expected {
		t.Errorf("Expected %v, got %v", expected, chats[0])
	}

	// Test case for non-existent chat
	filters = []FilterMemory{
		{Subject: "subject3"},
	}
	chats, err = am.RetrieveChatsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving chats by filter: %v", err)
	}

	if len(chats) != 0 {
		t.Errorf("Expected 0 chats, got %d", len(chats))
	}

	// Add more test cases for different scenarios
}

func TestRetrieveEventsByVector(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddEvent(created, &expiration, "subject1", "predicate1", "object1", "description1", []string{"keyword1", "keyword2"}, 0.5, "embedding_id1", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding event: %v", err)
	}

	err = am.AddEvent(created, &expiration, "subject2", "predicate2", "object2", "description2", []string{"keyword3", "keyword4"}, 0.7, "embedding_id2", []string{"filling3", "filling4"})
	if err != nil {
		t.Errorf("Error adding event: %v", err)
	}

	queryEmbedding, err := am.embedder.Embed("description1")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	events, err := am.RetrieveEventsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving events by vector: %v", err)
	}

	if len(events) != 1 {
		t.Errorf("Expected 1 event, got %d", len(events))
	}

	expected := ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if events[0] != expected {
		t.Errorf("Expected %v, got %v", expected, events[0])
	}

	// Test case for non-existent event
	queryEmbedding, err = am.embedder.Embed("description3")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	events, err = am.RetrieveEventsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving events by vector: %v", err)
	}

	if len(events) != 0 {
		t.Errorf("Expected 0 events, got %d", len(events))
	}

	// Test case for multiple events
	queryEmbedding, err = am.embedder.Embed("event")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	events, err = am.RetrieveEventsByVector(queryEmbedding, 10)
	if err != nil {
		t.Errorf("Error retrieving events by vector: %v", err)
	}

	if len(events) != 2 {
		t.Errorf("Expected 2 events, got %d", len(events))
	}

	// Add more test cases for different scenarios
}

func TestRetrieveThoughtsByVector(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddThought(created, &expiration, "subject1", "predicate1", "object1", "description1", []string{"keyword1", "keyword2"}, 0.5, "embedding_id1", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding thought: %v", err)
	}

	err = am.AddThought(created, &expiration, "subject2", "predicate2", "object2", "description2", []string{"keyword3", "keyword4"}, 0.7, "embedding_id2", []string{"filling3", "filling4"})
	if err != nil {
		t.Errorf("Error adding thought: %v", err)
	}

	queryEmbedding, err := am.embedder.Embed("description1")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	thoughts, err := am.RetrieveThoughtsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving thoughts by vector: %v", err)
	}

	if len(thoughts) != 1 {
		t.Errorf("Expected 1 thought, got %d", len(thoughts))
	}

	expected := ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if thoughts[0] != expected {
		t.Errorf("Expected %v, got %v", expected, thoughts[0])
	}

	// Test case for non-existent thought
	queryEmbedding, err = am.embedder.Embed("description3")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	thoughts, err = am.RetrieveThoughtsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving thoughts by vector: %v", err)
	}

	if len(thoughts) != 0 {
		t.Errorf("Expected 0 thoughts, got %d", len(thoughts))
	}

	// Test case for multiple thoughts
	queryEmbedding, err = am.embedder.Embed("thought")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	thoughts, err = am.RetrieveThoughtsByVector(queryEmbedding, 10)
	if err != nil {
		t.Errorf("Error retrieving thoughts by vector: %v", err)
	}

	if len(thoughts) != 2 {
		t.Errorf("Expected 2 thoughts, got %d", len(thoughts))
	}

	// Add more test cases for different scenarios
}

func TestRetrieveChatsByVector(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
	}
	logger := &log.Logger{}

	am, err := NewAssociativeMemory(cfg, logger)
	if err != nil {
		t.Errorf("Error creating AssociativeMemory: %v", err)
	}

	created := time.Now()
	expiration := created.Add(24 * time.Hour)
	err = am.AddChat(created, &expiration, "subject1", "predicate1", "object1", "description1", []string{"keyword1", "keyword2"}, 0.5, "embedding_id1", []string{"filling1", "filling2"})
	if err != nil {
		t.Errorf("Error adding chat: %v", err)
	}

	err = am.AddChat(created, &expiration, "subject2", "predicate2", "object2", "description2", []string{"keyword3", "keyword4"}, 0.7, "embedding_id2", []string{"filling3", "filling4"})
	if err != nil {
		t.Errorf("Error adding chat: %v", err)
	}

	queryEmbedding, err := am.embedder.Embed("description1")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	chats, err := am.RetrieveChatsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving chats by vector: %v", err)
	}

	if len(chats) != 1 {
		t.Errorf("Expected 1 chat, got %d", len(chats))
	}

	expected := ConceptNode{
		Subject:     "subject1",
		Predicate:   "predicate1",
		Object:      "object1",
		Description: "description1",
		Keywords:    []string{"keyword1", "keyword2"},
		Poignancy:   0.5,
		EmbeddingID: "embedding_id1",
		Filling:     []string{"filling1", "filling2"},
	}
	if chats[0] != expected {
		t.Errorf("Expected %v, got %v", expected, chats[0])
	}

	// Test case for non-existent chat
	queryEmbedding, err = am.embedder.Embed("description3")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	chats, err = am.RetrieveChatsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving chats by vector: %v", err)
	}

	if len(chats) != 0 {
		t.Errorf("Expected 0 chats, got %d", len(chats))
	}

	// Test case for multiple chats
	queryEmbedding, err = am.embedder.Embed("chat")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	chats, err = am.RetrieveChatsByVector(queryEmbedding, 10)
	if err != nil {
		t.Errorf("Error retrieving chats by vector: %v", err)
	}

	if len(chats) != 2 {
		t.Errorf("Expected 2 chats, got %d", len(chats))
	}

	// Add more test cases for different scenarios
}
