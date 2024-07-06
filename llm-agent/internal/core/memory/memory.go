package memory

import (
    "database/sql"
    "encoding/json"
    "fmt"
    "sync"
    "time"

    "github.com/lib/pq"
	"llm-agent/internal/types"
    "llm-agent/configs"
    "llm-agent/internal/interfaces"
    "llm-agent/pkg/utils"
    "github.com/rankbm25/bm25"
)


// MemoryCache is an interface for the caching layer
type MemoryCache interface {
    // Define methods for interacting with the caching layer
}

// MemoryStorage is an interface for the persistent storage layer
type MemoryStorage interface {
    // Define methods for interacting with the persistent storage
}

// CoreMemory is the central hub for managing the agent's memory components
type CoreMemory struct {
    serviceLocator *servicelocator.ServiceLocator
    storage        MemoryStorage
    cache          MemoryCache
    associative    *AssociativeMemory
    logger         *log.Logger
    mu             sync.RWMutex
}

// NewCoreMemory creates a new instance of CoreMemory
func NewCoreMemory(serviceLocator *servicelocator.ServiceLocator, logFile string) (*CoreMemory, error) {
    logWriter, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
    if err != nil {
        return nil, err
    }

    logger := log.New(logWriter, "CoreMemory: ", log.LstdFlags|log.Lmicroseconds|log.Lshortfile)

    storageService, err := serviceLocator.Get(reflect.TypeOf((*MemoryStorage)(nil)).Elem(), "storage")
    if err != nil {
        return nil, err
    }
    storage, ok := storageService.(MemoryStorage)
    if !ok {
        return nil, fmt.Errorf("service is not a MemoryStorage")
    }

    cacheService, err := serviceLocator.Get(reflect.TypeOf((*MemoryCache)(nil)).Elem(), "cache")
    if err != nil {
        // Handle the case where the cache service is not available
        cache = nil
    } else {
        cache, ok = cacheService.(MemoryCache)
        if !ok {
            return nil, fmt.Errorf("service is not a MemoryCache")
        }
    }

    associativeService, err := serviceLocator.Get(reflect.TypeOf((*AssociativeMemory)(nil)).Elem(), "associative")
    if err != nil {
        return nil, err
    }
    associative, ok := associativeService.(*AssociativeMemory)
    if !ok {
        return nil, fmt.Errorf("service is not an AssociativeMemory")
    }

    return &CoreMemory{
        serviceLocator: serviceLocator,
        storage:        storage,
        cache:          cache,
        associative:    associative,
        logger:         logger,
    }, nil
}

// AddEvent adds a new event to the associative memory
func (cm *CoreMemory) AddEvent(created time.Time, expiration *time.Time, subject, predicate, object, description string, keywords []string, poignancy float32, embeddingID string, filling []string) error {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    err := cm.associative.AddEvent(created, expiration, subject, predicate, object, description, keywords, poignancy, embeddingID, filling)
    if err != nil {
        cm.logger.Printf("Error adding event: %v", err)
        return err
    }

    return nil
}

// AddThought adds a new thought to the associative memory
func (cm *CoreMemory) AddThought(created time.Time, expiration *time.Time, subject, predicate, object, description string, keywords []string, poignancy float32, embeddingID string, filling []string) error {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    err := cm.associative.AddThought(created, expiration, subject, predicate, object, description, keywords, poignancy, embeddingID, filling)
    if err != nil {
        cm.logger.Printf("Error adding thought: %v", err)
        return err
    }

    return nil
}

// AddChat adds a new chat to the associative memory
func (cm *CoreMemory) AddChat(created time.Time, expiration *time.Time, subject, predicate, object, description string, keywords []string, poignancy float32, embeddingID string, filling []string) error {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    err := cm.associative.AddChat(created, expiration, subject, predicate, object, description, keywords, poignancy, embeddingID, filling)
    if err != nil {
        cm.logger.Printf("Error adding chat: %v", err)
        return err
    }

    return nil
}

// RetrieveEvents retrieves events from the associative memory based on filters
func (cm *CoreMemory) RetrieveEvents(filters ...Filter) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveEvents(filters...)
    if err != nil {
        cm.logger.Printf("Error retrieving events: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveThoughts retrieves thoughts from the associative memory based on filters
func (cm *CoreMemory) RetrieveThoughts(filters ...Filter) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveThoughts(filters...)
    if err != nil {
        cm.logger.Printf("Error retrieving thoughts: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveChats retrieves chats from the associative memory based on filters
func (cm *CoreMemory) RetrieveChats(filters ...Filter) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveChats(filters...)
    if err != nil {
        cm.logger.Printf("Error retrieving chats: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveEventsByVector retrieves events from the associative memory based on vector similarity
func (cm *CoreMemory) RetrieveEventsByVector(queryEmbedding []float32, k int) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveEventsByVector(queryEmbedding, k)
    if err != nil {
        cm.logger.Printf("Error retrieving events by vector: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveThoughtsByVector retrieves thoughts from the associative memory based on vector similarity
func (cm *CoreMemory) RetrieveThoughtsByVector(queryEmbedding []float32, k int) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveThoughtsByVector(queryEmbedding, k)
    if err != nil {
        cm.logger.Printf("Error retrieving thoughts by vector: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveChatsByVector retrieves chats from the associative memory based on vector similarity
func (cm *CoreMemory) RetrieveChatsByVector(queryEmbedding []float32, k int) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveChatsByVector(queryEmbedding, k)
    if err != nil {
        cm.logger.Printf("Error retrieving chats by vector: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveEventsByPlaintext retrieves events from the associative memory based on plaintext search
func (cm *CoreMemory) RetrieveEventsByPlaintext(query string, k int) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveEventsByPlaintext(query, k)
    if err != nil {
        cm.logger.Printf("Error retrieving events by plaintext: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveThoughtsByPlaintext retrieves thoughts from the associative memory based on plaintext search
func (cm *CoreMemory) RetrieveThoughtsByPlaintext(query string, k int) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveThoughtsByPlaintext(query, k)
    if err != nil {
        cm.logger.Printf("Error retrieving thoughts by plaintext: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveChatsByPlaintext retrieves chats from the associative memory based on plaintext search
func (cm *CoreMemory) RetrieveChatsByPlaintext(query string, k int) ([]ConceptNode, error) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    nodes, err := cm.associative.RetrieveChatsByPlaintext(query, k)
    if err != nil {
        cm.logger.Printf("Error retrieving chats by plaintext: %v", err)
        return nil, err
    }

    return nodes, nil
}

// RetrieveAndRerankEvents retrieves and reranks events from the associative memory
func (cm *CoreMemory) RetrieveAndRerankEvents(queryEmbedding []float32, queryText string, k int) ([]ConceptNode, error) {
    vectorResults, err := cm.RetrieveEventsByVector(queryEmbedding, k)
    if err != nil {
        return nil, err
    }

    plaintextResults, err := cm.RetrieveEventsByPlaintext(queryText, k)
    if err != nil {
        return nil, err
    }

    // Rerank the results using a custom reranking strategy
    rerankedResults := cm.rerank(vectorResults, plaintextResults)

    return rerankedResults, nil
}

// RetrieveAndRerankThoughts retrieves and reranks thoughts from the associative memory
func (cm *CoreMemory) RetrieveAndRerankThoughts(queryEmbedding []float32, queryText string, k int) ([]ConceptNode, error) {
    vectorResults, err := cm.RetrieveThoughtsByVector(queryEmbedding, k)
    if err != nil {
        return nil, err
    }

    plaintextResults, err := cm.RetrieveThoughtsByPlaintext(queryText, k)
    if err != nil {
        return nil, err
    }

    // Rerank the results using a custom reranking strategy
    rerankedResults := cm.rerank(vectorResults, plaintextResults)

    return rerankedResults, nil
}

// RetrieveAndRerankChats retrieves and reranks chats from the associative memory
func (cm *CoreMemory) RetrieveAndRerankChats(queryEmbedding []float32, queryText string, k int) ([]ConceptNode, error) {
    vectorResults, err := cm.RetrieveChatsByVector(queryEmbedding, k)
    if err != nil {
        return nil, err
    }

    plaintextResults, err := cm.RetrieveChatsByPlaintext(queryText, k)
    if err != nil {
        return nil, err
    }

    // Rerank the results using a custom reranking strategy
    rerankedResults := cm.rerank(vectorResults, plaintextResults)

    return rerankedResults, nil
}

// rerank is a placeholder function for reranking the results
func (cm *CoreMemory) rerank(vectorResults, plaintextResults []ConceptNode) []ConceptNode {
    // Implement your reranking strategy here
    // This could involve techniques like BM25, Reciprocal Rank Fusion, or other ensemble methods
    // For now, we'll return a dummy result
    return []ConceptNode{}
}

// SomeMethod is an example method for tracing and profiling
//func (cm *CoreMemory) SomeMethod() error {
//      tracePath := fmt.Sprintf("/tmp/trace-%d.out", time.Now().UnixNano())
//      traceFile, err := os.Create(tracePath)
//      if err != nil {
//              return err
//      }
//      defer traceFile.Close()
//
//      if err := trace.Start(traceFile); err != nil {
//              return err
//      }
//      defer trace.Stop()
//
//      cm.logger.Printf("Entering SomeMethod")
//      defer cm.logger.Printf("Exiting SomeMethod")
//
//      cpuProfilePath := fmt.Sprintf("/tmp/cpu-%d.pprof", time.Now().UnixNano())
//      cpuProfileFile, err := os.Create(cpuProfilePath)
//      if err != nil {
//              return err
//      }
//      defer cpuProfileFile.Close()
//
//      if err := pprof.StartCPUProfile(cpuProfileFile); err != nil {
//              return err
//      }
//      defer pprof.StopCPUProfile()
//
//      // Method implementation
//      // ...
//
//      return nil
//}
