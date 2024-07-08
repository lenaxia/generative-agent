package spatialmemory

import (
	"testing"
	"time"

	"llm-agent/configs"
	"llm-agent/internal/core/types"
	"llm-agent/pkg/embeddings"
	"llm-agent/pkg/vectors"
)

func TestAddLocation(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
		VectorSimilarityConfig: vectors.Config{
			Model: "cosine",
		},
	}
	logger := &log.Logger{}

	mt, err := NewMemoryTree(cfg, logger)
	if err != nil {
		t.Errorf("Error creating MemoryTree: %v", err)
	}

	err = mt.AddLocation("world1", "sector1", "arena1", []string{"obj1", "obj2"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	// Add more test cases for different scenarios
}

func TestGetStrAccessibleSectors(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
		VectorSimilarityConfig: vectors.Config{
			Model: "cosine",
		},
	}
	logger := &log.Logger{}

	mt, err := NewMemoryTree(cfg, logger)
	if err != nil {
		t.Errorf("Error creating MemoryTree: %v", err)
	}

	err = mt.AddLocation("world1", "sector1", "arena1", []string{"obj1", "obj2"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	err = mt.AddLocation("world1", "sector2", "arena2", []string{"obj3", "obj4"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	sectors, err := mt.GetStrAccessibleSectors("world1")
	if err != nil {
		t.Errorf("Error getting accessible sectors: %v", err)
	}

	expected := `["sector1", "sector2"]`
	if sectors != expected {
		t.Errorf("Expected %s, got %s", expected, sectors)
	}

	// Add more test cases for different scenarios
}

func TestGetStrAccessibleArenas(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
		VectorSimilarityConfig: vectors.Config{
			Model: "cosine",
		},
	}
	logger := &log.Logger{}

	mt, err := NewMemoryTree(cfg, logger)
	if err != nil {
		t.Errorf("Error creating MemoryTree: %v", err)
	}

	err = mt.AddLocation("world1", "sector1", "arena1", []string{"obj1", "obj2"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	err = mt.AddLocation("world1", "sector1", "arena2", []string{"obj3", "obj4"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	arenas, err := mt.GetStrAccessibleArenas("world1:sector1")
	if err != nil {
		t.Errorf("Error getting accessible arenas: %v", err)
	}

	expected := `["arena1", "arena2"]`
	if arenas != expected {
		t.Errorf("Expected %s, got %s", expected, arenas)
	}

	// Test case for non-existent sector
	arenas, err = mt.GetStrAccessibleArenas("world1:sector3")
	if err != nil {
		t.Errorf("Error getting accessible arenas: %v", err)
	}

	expected = `[]`
	if arenas != expected {
		t.Errorf("Expected %s, got %s", expected, arenas)
	}

	// Add more test cases for different scenarios
}

func TestGetStrAccessibleObjects(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
		VectorSimilarityConfig: vectors.Config{
			Model: "cosine",
		},
	}
	logger := &log.Logger{}

	mt, err := NewMemoryTree(cfg, logger)
	if err != nil {
		t.Errorf("Error creating MemoryTree: %v", err)
	}

	err = mt.AddLocation("world1", "sector1", "arena1", []string{"obj1", "obj2"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	objects, err := mt.GetStrAccessibleObjects("world1:sector1:arena1")
	if err != nil {
		t.Errorf("Error getting accessible objects: %v", err)
	}

	expected := `["obj1", "obj2"]`
	if objects != expected {
		t.Errorf("Expected %s, got %s", expected, objects)
	}

	// Test case for non-existent arena
	objects, err = mt.GetStrAccessibleObjects("world1:sector1:arena3")
	if err != nil {
		t.Errorf("Error getting accessible objects: %v", err)
	}

	expected = `[]`
	if objects != expected {
		t.Errorf("Expected %s, got %s", expected, objects)
	}

	// Add more test cases for different scenarios
}

func TestRetrieveLocationsByFilter(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
		VectorSimilarityConfig: vectors.Config{
			Model: "cosine",
		},
	}
	logger := &log.Logger{}

	mt, err := NewMemoryTree(cfg, logger)
	if err != nil {
		t.Errorf("Error creating MemoryTree: %v", err)
	}

	err = mt.AddLocation("world1", "sector1", "arena1", []string{"obj1", "obj2"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	err = mt.AddLocation("world1", "sector2", "arena2", []string{"obj3", "obj4"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	filters := []FilterLocation{
		{World: "world1", Sector: "sector1"},
	}
	locations, err := mt.RetrieveLocationsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving locations by filter: %v", err)
	}

	if len(locations) != 1 {
		t.Errorf("Expected 1 location, got %d", len(locations))
	}

	expected := Location{
		World:   "world1",
		Sector:  "sector1",
		Arena:   "arena1",
		Objects: []string{"obj1", "obj2"},
	}
	if locations[0] != expected {
		t.Errorf("Expected %v, got %v", expected, locations[0])
	}

	// Test case for multiple filters
	filters = []FilterLocation{
		{World: "world1", Sector: "sector1", Arena: "arena1", Objects: []string{"obj1"}},
	}
	locations, err = mt.RetrieveLocationsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving locations by filter: %v", err)
	}

	if len(locations) != 1 {
		t.Errorf("Expected 1 location, got %d", len(locations))
	}

	expected = Location{
		World:   "world1",
		Sector:  "sector1",
		Arena:   "arena1",
		Objects: []string{"obj1", "obj2"},
	}
	if locations[0] != expected {
		t.Errorf("Expected %v, got %v", expected, locations[0])
	}

	// Test case for non-existent location
	filters = []FilterLocation{
		{World: "world2", Sector: "sector3", Arena: "arena4"},
	}
	locations, err = mt.RetrieveLocationsByFilter(filters...)
	if err != nil {
		t.Errorf("Error retrieving locations by filter: %v", err)
	}

	if len(locations) != 0 {
		t.Errorf("Expected 0 locations, got %d", len(locations))
	}

	// Add more test cases for different scenarios
}

func TestRetrieveLocationsByVector(t *testing.T) {
	cfg := &configs.Config{
		DatabaseURL: "postgres://user:password@localhost/test_db",
		EmbeddingConfig: embeddings.Config{
			Model: "text-embedding-ada-002",
		},
		VectorSimilarityConfig: vectors.Config{
			Model: "cosine",
		},
	}
	logger := &log.Logger{}

	mt, err := NewMemoryTree(cfg, logger)
	if err != nil {
		t.Errorf("Error creating MemoryTree: %v", err)
	}

	err = mt.AddLocation("world1", "sector1", "arena1", []string{"obj1", "obj2"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	err = mt.AddLocation("world1", "sector2", "arena2", []string{"obj3", "obj4"})
	if err != nil {
		t.Errorf("Error adding location: %v", err)
	}

	queryEmbedding, err := mt.embedder.Embed("world1:sector1:arena1")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	locations, err := mt.RetrieveLocationsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving locations by vector: %v", err)
	}

	if len(locations) != 1 {
		t.Errorf("Expected 1 location, got %d", len(locations))
	}

	expected := Location{
		World:   "world1",
		Sector:  "sector1",
		Arena:   "arena1",
		Objects: []string{"obj1", "obj2"},
	}
	if locations[0] != expected {
		t.Errorf("Expected %v, got %v", expected, locations[0])
	}

	// Test case for non-existent location
	queryEmbedding, err = mt.embedder.Embed("world2:sector3:arena4")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	locations, err = mt.RetrieveLocationsByVector(queryEmbedding, 1)
	if err != nil {
		t.Errorf("Error retrieving locations by vector: %v", err)
	}

	if len(locations) != 0 {
		t.Errorf("Expected 0 locations, got %d", len(locations))
	}

	// Test case for multiple locations
	queryEmbedding, err = mt.embedder.Embed("world1")
	if err != nil {
		t.Errorf("Error generating embedding: %v", err)
	}

	locations, err = mt.RetrieveLocationsByVector(queryEmbedding, 10)
	if err != nil {
		t.Errorf("Error retrieving locations by vector: %v", err)
	}

	if len(locations) != 2 {
		t.Errorf("Expected 2 locations, got %d", len(locations))
	}

	// Add more test cases for different scenarios
}
