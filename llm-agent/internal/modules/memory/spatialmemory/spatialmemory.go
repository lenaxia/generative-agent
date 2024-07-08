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
	"llm-agent/internal/core/types"
	"llm-agent/pkg/embeddings"
	"llm-agent/pkg/utils"
	"llm-agent/pkg/vectors"
	"github.com/rankbm25/bm25"
)

// MemoryTree represents the agent's spatial memory
type MemoryTree struct {
	db              *sql.DB
	logger          *log.Logger
	mu              sync.RWMutex
	embedder        embeddings.Embedder
	vectorSimilarity vectors.VectorSimilarity
	bm25            *bm25.BM25
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
			objects TEXT[],
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

	mt := &MemoryTree{
		db:              db,
		logger:          logger,
		embedder:        embedder,
		vectorSimilarity: vectorSimilarity,
	}
	mt.initBM25()

	return mt, nil
}

// GetMetadata returns the metadata for this service
func (mt *MemoryTree) GetMetadata() interfaces.ServiceMetadata {
	return interfaces.ServiceMetadata{
		Description: "Manages the agent's spatial memory",
		Metadata:    make(map[string]interface{}),
	}
}

// AddLocation adds a new location to the memory tree
func (mt *MemoryTree) AddLocation(world, sector, arena string, objects []string) error {
	mt.mu.Lock()
	defer mt.mu.Unlock()

	locationStr := fmt.Sprintf("%s:%s:%s", world, sector, arena)
	embedding, err := mt.embedder.Embed(locationStr)
	if err != nil {
		mt.logger.Printf("Error generating embedding for location %s: %v", locationStr, err)
		return err
	}

	_, err = mt.db.Exec(`
		INSERT INTO memory_tree (world, sector, arena, objects, embedding)
		VALUES ($1, $2, $3, $4, $5)
	`, world, sector, arena, pq.Array(objects), pq.Array(embedding))
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

// GetStrAccessibleObjects returns a string of accessible objects in the current arena
func (mt *MemoryTree) GetStrAccessibleObjects(arena string) (string, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	currWorld, currSector, currArena := splitArena(arena)
	if currArena == "" {
		return "", nil
	}

	var objects []string
	err := mt.db.QueryRow(`
		SELECT objects
		FROM memory_tree
		WHERE world = $1 AND sector = $2 AND arena = $3
	`, currWorld, currSector, currArena).Scan(pq.Array(&objects))
	if err != nil {
		mt.logger.Printf("Error retrieving accessible objects: %v", err)
		return "", err
	}

	return fmt.Sprintf("%q", objects), nil
}

// GetFilterString returns a string representation of the filter
func (f FilterLocation) GetFilterString() string {
    filterStr := ""
    if f.World != "" {
        filterStr += "World: " + f.World + " "
    }
    if f.Sector != "" {
        filterStr += "Sector: " + f.Sector + " "
    }
    if f.Arena != "" {
        filterStr += "Arena: " + f.Arena + " "
    }
    if len(f.Objects) > 0 {
        filterStr += "Objects: " + strings.Join(f.Objects, ", ") + " "
    }
    if f.PlaintextQuery != "" {
        filterStr += "PlaintextQuery: " + f.PlaintextQuery + " "
    }
    return strings.TrimSpace(filterStr)
}

// RetrieveLocations retrieves locations from the memory tree based on filters and vector similarity
func (cm *CoreMemory) RetrieveLocations(filters ...FilterLocation) (*SearchResults, error) {
    var queryEmbedding []float32
    for _, filter := range filters {
        filterStr := filter.GetFilterString()
        if filterStr != "" {
            embedding, err := cm.embedder.Embed(filterStr)
            if err != nil {
                return nil, err
            }
            queryEmbedding = embedding
            break
        }
    }

    plaintextResults, err := cm.spatial.RetrieveLocationsByFilter(filters...)
    if err != nil {
        return nil, err
    }

    vectorResults, err := cm.spatial.RetrieveLocationsByVector(queryEmbedding, len(plaintextResults))
    if err != nil {
        return nil, err
    }

    return &SearchResults{
        PlaintextResults: plaintextResults,
        VectorResults:    vectorResults,
    }, nil
}

// RetrieveLocationsByFilter retrieves locations from the memory tree based on filters
func (mt *MemoryTree) RetrieveLocationsByFilter(filters ...FilterLocation) ([]Location, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	var locations []Location
	query := `SELECT world, sector, arena, objects, embedding FROM memory_tree`

	// Apply filters to the query
	for _, filter := range filters {
		if filter.World != "" {
			query += fmt.Sprintf(" AND world = '%s'", filter.World)
		}
		if filter.Sector != "" {
			query += fmt.Sprintf(" AND sector = '%s'", filter.Sector)
		}
		if filter.Arena != "" {
			query += fmt.Sprintf(" AND arena = '%s'", filter.Arena)
		}
		if len(filter.Objects) > 0 {
			query += fmt.Sprintf(" AND objects && %s", pq.Array(filter.Objects))
		}
		if filter.PlaintextQuery != "" {
			queryTokens := utils.Tokenize(filter.PlaintextQuery)
			scores := mt.bm25.ScoreTokens(queryTokens)
			query += fmt.Sprintf(" AND (SELECT SUM(bm25_score(keywords, %s, %s)) FROM unnest(keywords) AS keyword) > 0", pq.Array(queryTokens), pq.Array(scores))
		}
		if len(filter.EmbeddingIDs) > 0 {
			query += fmt.Sprintf(" AND embedding_id IN (%s)", pq.Array(filter.EmbeddingIDs))
		}
	}

	rows, err := mt.db.Query(query)
	if err != nil {
		mt.logger.Printf("Error retrieving locations: %v", err)
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var location Location
		var embedding []float32
		err = rows.Scan(&location.World, &location.Sector, &location.Arena, pq.Array(&location.Objects), pq.Array(&embedding))
		if err != nil {
			mt.logger.Printf("Error scanning location: %v", err)
			return nil, err
		}
		location.Embedding = embedding
		locations = append(locations, location)
	}

	return locations, nil
}

// RetrieveLocationsByVector retrieves locations from the memory tree based on vector similarity
func (mt *MemoryTree) RetrieveLocationsByVector(queryEmbedding []float32, k int) ([]Location, error) {
	mt.mu.RLock()
	defer mt.mu.RUnlock()

	var locations []Location
	query := `
		SELECT world, sector, arena, objects, embedding
		FROM memory_tree
		ORDER BY embedding <-> $1 ASC
		LIMIT $2
	`

	rows, err := mt.db.Query(query, pq.Array(queryEmbedding), k)
	if err != nil {
		mt.logger.Printf("Error retrieving locations by vector: %v", err)
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var location Location
		var embedding []float32
		err = rows.Scan(&location.World, &location.Sector, &location.Arena, pq.Array(&location.Objects), pq.Array(&embedding))
		if err != nil {
			mt.logger.Printf("Error scanning location: %v", err)
			return nil, err
		}
		location.Embedding = embedding
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

func (mt *MemoryTree) initBM25() {
	var corpus []string
	rows, err := mt.db.Query(`SELECT world || ':' || sector || ':' || arena FROM memory_tree`)
	if err != nil {
		mt.logger.Printf("Error retrieving corpus for BM25: %v", err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var location string
		err = rows.Scan(&location)
		if err != nil {
			mt.logger.Printf("Error scanning location for BM25: %v", err)
			return
		}
		corpus = append(corpus, location)
	}

	mt.bm25 = bm25.NewBM25(corpus)
}

// deduplicateLocations removes duplicate locations from a slice
func deduplicateLocations(locations []Location) []Location {
    seen := make(map[Location]struct{}, len(locations))
    result := make([]Location, 0, len(locations))
    for _, location := range locations {
        if _, ok := seen[location]; !ok {
            seen[location] = struct{}{}
            result = append(result, location)
        }
    }
    return result
}
