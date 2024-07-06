package spatialmemory

import (
	"database/sql"
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/lib/pq"
	"llm-agent/configs"
	"llm-agent/internal/interfaces"
	"llm-agent/pkg/embeddings"
	"llm-agent/pkg/vectors"
)

// MemoryTree represents the agent's spatial memory
type MemoryTree struct {
	db              *sql.DB
	logger          *log.Logger
	mu              sync.RWMutex
	embedder        embeddings.Embedder
	vectorSimilarity vectors.VectorSimilarity
}

// NewMemoryTree creates a new instance of MemoryTree
func NewMemoryTree(cfg *configs.Config, logger *log.Logger) (*MemoryTree, error) {
	db, err := sql.Open("postgres", cfg.DatabaseURL)
	if err != nil {
		return nil, err
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS memory_tree (
			id SERIAL PRIMARY KEY,
			world TEXT NOT NULL,
			sector TEXT NOT NULL,
			arena TEXT NOT NULL,
			game_objects TEXT[],
			embedding VECTOR(512)
		);
	`)
	if err != nil {
		return nil, err
	}

	embedder, err := embeddings.NewEmbedder(cfg.EmbeddingConfig)
	if err != nil {
		return nil, err
	}

	vectorSimilarity, err := vectors.NewVectorSimilarity(cfg.VectorSimilarityConfig)
	if err != nil {
		return nil, err
	}

	return &MemoryTree{
		db:              db,
		logger:          logger,
		embedder:        embedder,
		vectorSimilarity: vectorSimilarity,
	}, nil
}

// GetMetadata returns the metadata for this service
func (mt *MemoryTree) GetMetadata() interfaces.ServiceMetadata {
	return interfaces.ServiceMetadata{
		Description: "Manages the agent's spatial memory",
		Metadata:    make(map[string]interface{}),
	}
}

// AddLocation adds a new location to the memory tree
func (mt *MemoryTree) AddLocation(world, sector, arena string, gameObjects []string) error {
	mt.mu.Lock()
	defer mt.mu.Unlock()

	locationStr := fmt.Sprintf("%s:%s:%s", world, sector, arena)
	embedding, err := mt.embedder.Embed(locationStr)
	if err != nil {
		mt.logger.Printf("Error generating embedding for location %s: %v", locationStr, err)
		return err
	}

	_, err = mt.db.Exec(`
		INSERT INTO memory_tree (world, sector, arena, game_objects, embedding)
		VALUES ($1, $2, $3, $4, $5)
	`, world, sector, arena, pq.Array(gameObjects), pq.Array(embedding))
	if err != nil {
		mt.logger.Printf("Error inserting location %s: %v", locationStr, err)
		return err
	}

	mt.logger.Printf("Added location %s to memory tree", locationStr)
	return nil
}

// GetStrAccessibleSectors returns a string of accessible sectors in the current world
func (mt *MemoryTree) GetStrAccessibleSectors(currWorld string) (string, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	var sectors []string
	err := mt.db.QueryRow(`
		SELECT array_agg(sector)
		FROM memory_tree
		WHERE world = $1
		GROUP BY world
	`, currWorld).Scan(pq.Array(&sectors))
	if err != nil {
		mt.logger.Printf("Error retrieving accessible sectors: %v", err)
		return "", err
	}

	return fmt.Sprintf("%q", sectors), nil
}

// GetStrAccessibleArenas returns a string of accessible arenas in the current sector
func (mt *MemoryTree) GetStrAccessibleArenas(sector string) (string, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	currWorld, currSector := splitSector(sector)
	if currSector == "" {
		return "", nil
	}

	var arenas []string
	err := mt.db.QueryRow(`
		SELECT array_agg(arena)
		FROM memory_tree
		WHERE world = $1 AND sector = $2
		GROUP BY world, sector
	`, currWorld, currSector).Scan(pq.Array(&arenas))
	if err != nil {
		mt.logger.Printf("Error retrieving accessible arenas: %v", err)
		return "", err
	}

	return fmt.Sprintf("%q", arenas), nil
}

// GetStrAccessibleGameObjects returns a string of accessible game objects in the current arena
func (mt *MemoryTree) GetStrAccessibleGameObjects(arena string) (string, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	currWorld, currSector, currArena := splitArena(arena)
	if currArena == "" {
		return "", nil
	}

	var gameObjects []string
	err := mt.db.QueryRow(`
		SELECT game_objects
		FROM memory_tree
		WHERE world = $1 AND sector = $2 AND arena = $3
	`, currWorld, currSector, currArena).Scan(pq.Array(&gameObjects))
	if err != nil {
		mt.logger.Printf("Error retrieving accessible game objects: %v", err)
		return "", err
	}

	return fmt.Sprintf("%q", gameObjects), nil
}

// RetrieveSimilarLocations retrieves locations similar to the given query string
func (mt *MemoryTree) RetrieveSimilarLocations(query string, k int) ([]string, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	queryEmbedding, err := mt.embedder.Embed(query)
	if err != nil {
		mt.logger.Printf("Error generating embedding for query %s: %v", query, err)
		return nil, err
	}

	var locations []string
	rows, err := mt.db.Query(`
		SELECT world || ':' || sector || ':' || arena
		FROM memory_tree
		ORDER BY embedding <-> $1 ASC
		LIMIT $2
	`, pq.Array(queryEmbedding), k)
	if err != nil {
		mt.logger.Printf("Error retrieving similar locations: %v", err)
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var location string
		err = rows.Scan(&location)
		if err != nil {
			mt.logger.Printf("Error scanning location: %v", err)
			return nil, err
		}
		locations = append(locations, location)
	}

	return locations, nil
}

// splitSector splits a sector string into world and sector components
func splitSector(sector string) (string, string) {
	parts := strings.Split(sector, ":")
	if len(parts) < 2 {
		return "", ""
	}
	return parts[0], parts[1]
}

// splitArena splits an arena string into world, sector, and arena components
func splitArena(arena string) (string, string, string) {
	parts := strings.Split(arena, ":")
	if len(parts) < 3 {
		return "", "", ""
	}
	return parts[0], parts[1], parts[2]
}
