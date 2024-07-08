package associativememory

import (
    "database/sql"
    "encoding/json"
    "fmt"
    "sync"
    "time"

    "github.com/lib/pq"
    "llm-agent/configs"
    "llm-agent/internal/interfaces"
    "llm-agent/internal/core/types"
    "llm-agent/pkg/utils"
    "github.com/rankbm25/bm25"
)

// AssociativeMemory is the module for managing the agent's long-term memory
type AssociativeMemory struct {
    db     *sql.DB
    mu     sync.RWMutex
    logger *log.Logger
    bm25   *bm25.BM25
}

// NewAssociativeMemory creates a new instance of AssociativeMemory
func NewAssociativeMemory(cfg *configs.Config, logger *log.Logger) (*AssociativeMemory, error) {
    db, err := sql.Open("postgres", cfg.DatabaseURL)
    if err != nil {
        return nil, err
    }

    _, err = db.Exec(`
        CREATE TABLE IF NOT EXISTS concept_nodes (
            id TEXT PRIMARY KEY,
            node_count INT NOT NULL,
            type_count INT NOT NULL,
            type TEXT NOT NULL,
            depth INT NOT NULL,
            created TIMESTAMP NOT NULL,
            expiration TIMESTAMP,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            description TEXT NOT NULL,
            embedding_id TEXT NOT NULL,
            poignancy REAL NOT NULL,
            keywords TEXT[] NOT NULL,
            filling TEXT[]
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            embedding VECTOR(512) NOT NULL
        );
    `)
    if err != nil {
        return nil, err
    }

    am := &AssociativeMemory{db: db, logger: logger}
    am.initBM25()

    return am, nil
}

// GetMetadata returns the metadata for this service
func (am *AssociativeMemory) GetMetadata() interfaces.ServiceMetadata {
    return interfaces.ServiceMetadata{
        Description: "Manages the agent's long-term memory",
        Metadata:    make(map[string]interface{}),
    }
}

// AddEvent adds a new event to the associative memory
func (am *AssociativeMemory) AddEvent(created time.Time, expiration *time.Time, subject, predicate, object, description string, keywords []string, poignancy float32, embeddingID string, filling []string) error {
    am.mu.Lock()
    defer am.mu.Unlock()

    node := ConceptNode{
        NodeCount:   0, // Will be updated by the database
        TypeCount:   0, // Will be updated by the database
        Type:        "event",
        Depth:       0,
        Created:     created,
        Expiration:  expiration,
        Subject:     subject,
        Predicate:   predicate,
        Object:      object,
        Description: description,
        EmbeddingID: embeddingID,
        Poignancy:   poignancy,
        Keywords:    pq.StringArray(keywords),
        Filling:     filling,
    }

    nodeBytes, err := json.Marshal(node)
    if err != nil {
        am.logger.Printf("Error marshaling node: %v", err)
        return err
    }

    _, err = am.db.Exec(`
        INSERT INTO concept_nodes (
            node_count, type_count, type, depth, created, expiration, subject, predicate, object, description, embedding_id, poignancy, keywords, filling
        ) VALUES (
            (SELECT COALESCE(MAX(node_count), 0) + 1 FROM concept_nodes),
            (SELECT COALESCE(MAX(type_count), 0) + 1 FROM concept_nodes WHERE type = 'event'),
            'event', 0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        ) RETURNING id
    `, created, pq.NullTime{Time: *expiration, Valid: expiration != nil}, subject, predicate, object, description, embeddingID, poignancy, pq.StringArray(keywords), pq.Array(filling))
    if err != nil {
        am.logger.Printf("Error inserting event node: %v", err)
        return err
    }

    am.logger.Printf("Added event node: %s", string(nodeBytes))
    return nil
}

// AddThought adds a new thought to the associative memory
func (am *AssociativeMemory) AddThought(created time.Time, expiration *time.Time, subject, predicate, object, description string, keywords []string, poignancy float32, embeddingID string, filling []string) error {
    am.mu.Lock()
    defer am.mu.Unlock()

    node := ConceptNode{
        NodeCount:   0, // Will be updated by the database
        TypeCount:   0, // Will be updated by the database
        Type:        "thought",
        Depth:       1, // Thoughts have a depth of 1
        Created:     created,
        Expiration:  expiration,
        Subject:     subject,
        Predicate:   predicate,
        Object:      object,
        Description: description,
        EmbeddingID: embeddingID,
        Poignancy:   poignancy,
        Keywords:    pq.StringArray(keywords),
        Filling:     filling,
    }

    // Calculate depth based on filling
    if len(filling) > 0 {
        var maxDepth int
        for _, nodeID := range filling {
            var depth int
            err := am.db.QueryRow(`SELECT depth FROM concept_nodes WHERE id = $1`, nodeID).Scan(&depth)
            if err != nil {
                am.logger.Printf("Error retrieving depth for node %s: %v", nodeID, err)
                return err
            }
            if depth > maxDepth {
                maxDepth = depth
            }
        }
        node.Depth = maxDepth + 1
    }

    nodeBytes, err := json.Marshal(node)
    if err != nil {
        am.logger.Printf("Error marshaling node: %v", err)
        return err
    }

    _, err = am.db.Exec(`
        INSERT INTO concept_nodes (
            node_count, type_count, type, depth, created, expiration, subject, predicate, object, description, embedding_id, poignancy, keywords, filling
        ) VALUES (
            (SELECT COALESCE(MAX(node_count), 0) + 1 FROM concept_nodes),
            (SELECT COALESCE(MAX(type_count), 0) + 1 FROM concept_nodes WHERE type = 'thought'),
            'thought', $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        ) RETURNING id
    `, node.Depth, created, pq.NullTime{Time: *expiration, Valid: expiration != nil}, subject, predicate,object, description, embeddingID, poignancy, pq.StringArray(keywords), pq.Array(filling))
    if err != nil {
        am.logger.Printf("Error inserting thought node: %v", err)
        return err
    }

    am.logger.Printf("Added thought node: %s", string(nodeBytes))
    return nil
}

// AddChat adds a new chat to the associative memory
func (am *AssociativeMemory) AddChat(created time.Time, expiration *time.Time, subject, predicate, object, description string, keywords []string, poignancy float32, embeddingID string, filling []string) error {
    am.mu.Lock()
    defer am.mu.Unlock()

    node := ConceptNode{
        NodeCount:   0, // Will be updated by the database
        TypeCount:   0, // Will be updated by the database
        Type:        "chat",
        Depth:       0, // Chats have a depth of 0
        Created:     created,
        Expiration:  expiration,
        Subject:     subject,
        Predicate:   predicate,
        Object:      object,
        Description: description,
        EmbeddingID: embeddingID,
        Poignancy:   poignancy,
        Keywords:    pq.StringArray(keywords),
        Filling:     filling,
    }

    nodeBytes, err := json.Marshal(node)
    if err != nil {
        am.logger.Printf("Error marshaling node: %v", err)
        return err
    }

    _, err = am.db.Exec(`
        INSERT INTO concept_nodes (
            node_count, type_count, type, depth, created, expiration, subject, predicate, object, description, embedding_id, poignancy, keywords, filling
        ) VALUES (
            (SELECT COALESCE(MAX(node_count), 0) + 1 FROM concept_nodes),
            (SELECT COALESCE(MAX(type_count), 0) + 1 FROM concept_nodes WHERE type = 'chat'),
            'chat', 0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        ) RETURNING id
    `, created, pq.NullTime{Time: *expiration, Valid: expiration != nil}, subject, predicate, object, description, embeddingID, poignancy, pq.StringArray(keywords), pq.Array(filling))
    if err != nil {
        am.logger.Printf("Error inserting chat node: %v", err)
        return err
    }

    am.logger.Printf("Added chat node: %s", string(nodeBytes))
    return nil
}

// RetrieveEventsByFilter retrieves events from the associative memory based on filters
func (am *AssociativeMemory) RetrieveEvents(filters ...FilterMemory) (*SearchResults, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    plaintextResults, err := am.RetrieveEventsByFilter(filters...)
    if err != nil {
        return nil, err
    }

    vectorResults, err := am.RetrieveEventsByVector(filters...)
    if err != nil {
        return nil, err
    }

    return &SearchResults{
        PlaintextResults: plaintextResults,
        VectorResults:    vectorResults,
    }, nil
}

// RetrieveThoughtsByFilter retrieves thoughts from the associative memory based on filters
func (am *AssociativeMemory) RetrieveThoughts(filters ...FilterMemory) (*SearchResults, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    plaintextResults, err := am.RetrieveThoughtsByFilter(filters...)
    if err != nil {
        return nil, err
    }

    vectorResults, err := am.RetrieveThoughtsByVector(filters...)
    if err != nil {
        return nil, err
    }

    return &SearchResults{
        PlaintextResults: plaintextResults,
        VectorResults:    vectorResults,
    }, nil
}

// RetrieveChatsByFilter retrieves chats from the associative memory based on filters
func (am *AssociativeMemory) RetrieveChats(filters ...FilterMemory) (*SearchResults, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    plaintextResults, err := am.RetrieveChatsByFilter(filters...)
    if err != nil {
        return nil, err
    }

    vectorResults, err := am.RetrieveChatsByVector(filters...)
    if err != nil {
        return nil, err
    }

    return &SearchResults{
        PlaintextResults: plaintextResults,
        VectorResults:    vectorResults,
    }, nil
}

// RetrieveEvents retrieves events from the associative memory based on filters
func (am *AssociativeMemory) RetrieveEventsByFilter(filters ...FilterMemory) ([]ConceptNode, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    var nodes []ConceptNode
    query := `SELECT id, node_count, type_count, type, depth, created, expiration, subject, predicate, object, description, embedding_id, poignancy, keywords, filling FROM concept_nodes WHERE type = 'event'`

    // Apply filters to the query
    for _, filter := range filters {
        if filter.NodeType != "" {
            query += fmt.Sprintf(" AND type = '%s'", filter.NodeType)
        }
        if filter.Subject != "" {
            query += fmt.Sprintf(" AND subject = '%s'", filter.Subject)
        }
        if filter.Predicate != "" {
            query += fmt.Sprintf(" AND predicate = '%s'", filter.Predicate)
        }
        if filter.Object != "" {
            query += fmt.Sprintf(" AND object = '%s'", filter.Object)
        }
        if len(filter.Keywords) > 0 {
            query += fmt.Sprintf(" AND keywords && %s", pq.Array(filter.Keywords))
        }
        if filter.PlaintextQuery != "" {
            queryTokens := utils.Tokenize(filter.PlaintextQuery)
            scores := am.bm25.ScoreTokens(queryTokens)
            query += fmt.Sprintf(" AND (SELECT SUM(bm25_score(keywords, %s, %s)) FROM unnest(keywords) AS keyword) > 0", pq.Array(queryTokens), pq.Array(scores))
        }
        if filter.StartTime != nil {
            query += fmt.Sprintf(" AND created >= '%s'", filter.StartTime.Format(time.RFC3339))
        }
        if filter.EndTime != nil {
            query += fmt.Sprintf(" AND created <= '%s'", filter.EndTime.Format(time.RFC3339))
        }
        if len(filter.EmbeddingIDs) > 0 {
            query += fmt.Sprintf(" AND embedding_id IN (%s)", pq.Array(filter.EmbeddingIDs))
        }
    }

    rows, err := am.db.Query(query)
    if err != nil {
        am.logger.Printf("Error retrieving events: %v", err)
        return nil, err
    }
    defer rows.Close()

    for rows.Next() {
        var node ConceptNode
        var expiration pq.NullTime
        err = rows.Scan(&node.ID, &node.NodeCount, &node.TypeCount, &node.Type, &node.Depth, &node.Created, &expiration, &node.Subject, &node.Predicate, &node.Object, &node.Description, &node.EmbeddingID, &node.Poignancy, pq.Array(&node.Keywords), pq.Array(&node.Filling))
        if err != nil {
            am.logger.Printf("Error scanning event node: %v", err)
            return nil, err
        }
        if expiration.Valid {
            node.Expiration = &expiration.Time
        }
        nodes = append(nodes, node)
    }

    return nodes, nil
}

// RetrieveThoughts retrieves thoughts from the associative memory based on filters
func (am *AssociativeMemory) RetrieveThoughtsByFilter(filters ...FilterMemory) ([]ConceptNode, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    var nodes []ConceptNode
    query := `SELECT id, node_count, type_count, type, depth, created, expiration, subject, predicate, object, description, embedding_id, poignancy, keywords, filling FROM concept_nodes WHERE type = 'thought'`

    // Apply filters to the query
    for _, filter := range filters {
        if filter.NodeType != "" {
            query += fmt.Sprintf(" AND type = '%s'", filter.NodeType)
        }
        if filter.Subject != "" {
            query += fmt.Sprintf(" AND subject = '%s'", filter.Subject)
        }
        if filter.Predicate != "" {
            query += fmt.Sprintf(" AND predicate = '%s'", filter.Predicate)
        }
        if filter.Object != "" {
            query += fmt.Sprintf(" AND object = '%s'", filter.Object)
        }
        if len(filter.Keywords) > 0 {
            query += fmt.Sprintf(" AND keywords && %s", pq.Array(filter.Keywords))
        }
        if filter.PlaintextQuery != "" {
            queryTokens := utils.Tokenize(filter.PlaintextQuery)
            scores := am.bm25.ScoreTokens(queryTokens)
            query += fmt.Sprintf(" AND (SELECT SUM(bm25_score(keywords, %s, %s)) FROM unnest(keywords) AS keyword) > 0", pq.Array(queryTokens), pq.Array(scores))
        }
        if filter.StartTime != nil {
            query += fmt.Sprintf(" AND created >= '%s'", filter.StartTime.Format(time.RFC3339))
        }
        if filter.EndTime != nil {
            query += fmt.Sprintf(" AND created <= '%s'", filter.EndTime.Format(time.RFC3339))
        }
        if len(filter.EmbeddingIDs) > 0 {
            query += fmt.Sprintf(" AND embedding_id IN (%s)", pq.Array(filter.EmbeddingIDs))
        }
    }

    rows, err := am.db.Query(query)
    if err != nil {
        am.logger.Printf("Error retrieving thoughts: %v", err)
        return nil, err
    }
    defer rows.Close()

    for rows.Next() {
        var node ConceptNode
        var expiration pq.NullTime
        err = rows.Scan(&node.ID, &node.NodeCount, &node.TypeCount, &node.Type, &node.Depth, &node.Created, &expiration, &node.Subject, &node.Predicate, &node.Object, &node.Description, &node.EmbeddingID, &node.Poignancy, pq.Array(&node.Keywords), pq.Array(&node.Filling))
        if err != nil {
            am.logger.Printf("Error scanning thought node: %v", err)
            return nil, err
        }
        if expiration.Valid {
            node.Expiration = &expiration.Time
        }
        nodes = append(nodes, node)
    }

    return nodes, nil
}

// RetrieveChats retrieves chats from the associative memory based on filters
func (am *AssociativeMemory) RetrieveChatsByFilter(filters ...FilterMemory) ([]ConceptNode, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    var nodes []ConceptNode
    query := `SELECT id, node_count, type_count, type, depth, created, expiration, subject, predicate, object, description, embedding_id, poignancy, keywords, filling FROM concept_nodes WHERE type = 'chat'`

    // Apply filters to the query
    for _, filter := range filters {
        if filter.NodeType != "" {
            query += fmt.Sprintf(" AND type = '%s'", filter.NodeType)
        }
        if filter.Subject != "" {
            query += fmt.Sprintf(" AND subject = '%s'", filter.Subject)
        }
        if filter.Predicate != "" {
            query += fmt.Sprintf(" AND predicate = '%s'", filter.Predicate)
        }
        if filter.Object != "" {
            query += fmt.Sprintf(" AND object = '%s'", filter.Object)
        }
        if len(filter.Keywords) > 0 {
            query += fmt.Sprintf(" AND keywords && %s", pq.Array(filter.Keywords))
        }
        if filter.PlaintextQuery != "" {
            queryTokens := utils.Tokenize(filter.PlaintextQuery)
            scores := am.bm25.ScoreTokens(queryTokens)
            query += fmt.Sprintf(" AND (SELECT SUM(bm25_score(keywords, %s, %s)) FROM unnest(keywords) AS keyword) > 0", pq.Array(queryTokens), pq.Array(scores))
        }
        if filter.StartTime != nil {
            query += fmt.Sprintf(" AND created >= '%s'", filter.StartTime.Format(time.RFC3339))
        }
        if filter.EndTime != nil {
            query += fmt.Sprintf(" AND created <= '%s'", filter.EndTime.Format(time.RFC3339))
        }
        if len(filter.EmbeddingIDs) > 0 {
            query += fmt.Sprintf(" AND embedding_id IN (%s)", pq.Array(filter.EmbeddingIDs))
        }
    }

    rows, err := am.db.Query(query)
    if err != nil {
        am.logger.Printf("Error retrieving chats: %v", err)
        return nil, err
    }
    defer rows.Close()

    for rows.Next() {
        var node ConceptNode
        var expiration pq.NullTime
        err = rows.Scan(&node.ID, &node.NodeCount, &node.TypeCount, &node.Type, &node.Depth, &node.Created, &expiration, &node.Subject, &node.Predicate, &node.Object, &node.Description, &node.EmbeddingID, &node.Poignancy, pq.Array(&node.Keywords), pq.Array(&node.Filling))
        if err != nil {
            am.logger.Printf("Error scanning chat node: %v", err)
            return nil, err
        }
        if expiration.Valid {
            node.Expiration = &expiration.Time
        }
        nodes = append(nodes, node)
    }

    return nodes, nil
}

// RetrieveEventsByVector retrieves events from the associative memory based on vector similarity
func (am *AssociativeMemory) RetrieveEventsByVector(queryEmbedding []float32, k int) ([]ConceptNode, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    var nodes []ConceptNode
    query := `
        SELECT cn.id, cn.node_count, cn.type_count, cn.type, cn.depth, cn.created, cn.expiration, cn.subject, cn.predicate, cn.object, cn.description, cn.embedding_id, cn.poignancy, cn.keywords, cn.filling
        FROM concept_nodes cn
        INNER JOIN embeddings e ON cn.embedding_id = e.id
        WHERE cn.type = 'event'
        ORDER BY e.embedding <-> $1 ASC
        LIMIT $2
    `

    rows, err := am.db.Query(query, pq.Array(queryEmbedding), k)
    if err != nil {
        am.logger.Printf("Error retrieving events by vector: %v", err)
        return nil, err
    }
    defer rows.Close()

    for rows.Next() {
        var node ConceptNode
        var expiration pq.NullTime
        err = rows.Scan(&node.ID, &node.NodeCount, &node.TypeCount, &node.Type, &node.Depth, &node.Created, &expiration, &node.Subject, &node.Predicate, &node.Object, &node.Description, &node.EmbeddingID, &node.Poignancy, pq.Array(&node.Keywords), pq.Array(&node.Filling))
        if err != nil {
            am.logger.Printf("Error scanning event node: %v", err)
            return nil, err
        }
        if expiration.Valid {
            node.Expiration = &expiration.Time
        }
        nodes = append(nodes, node)
    }

    return nodes, nil
}

// RetrieveThoughtsByVector retrieves thoughts from the associative memory based on vector similarity
func (am *AssociativeMemory) RetrieveThoughtsByVector(queryEmbedding []float32, k int) ([]ConceptNode, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    var nodes []ConceptNode
    query := `
        SELECT cn.id, cn.node_count, cn.type_count, cn.type, cn.depth, cn.created, cn.expiration, cn.subject, cn.predicate, cn.object, cn.description, cn.embedding_id, cn.poignancy, cn.keywords, cn.filling
        FROM concept_nodes cn
        INNER JOIN embeddings e ON cn.embedding_id = e.id
        WHERE cn.type = 'thought'
        ORDER BY e.embedding <-> $1 ASC
        LIMIT $2
    `

    rows, err := am.db.Query(query, pq.Array(queryEmbedding), k)
    if err != nil {
        am.logger.Printf("Error retrieving thoughts by vector: %v", err)
        return nil, err
    }
    defer rows.Close()

    for rows.Next() {
        var node ConceptNode
        var expiration pq.NullTime
        err = rows.Scan(&node.ID, &node.NodeCount, &node.TypeCount, &node.Type, &node.Depth, &node.Created, &expiration, &node.Subject, &node.Predicate, &node.Object, &node.Description, &node.EmbeddingID, &node.Poignancy, pq.Array(&node.Keywords), pq.Array(&node.Filling))
        if err != nil {
            am.logger.Printf("Error scanning thought node: %v", err)
            return nil, err
        }
        if expiration.Valid {
            node.Expiration = &expiration.Time
        }
        nodes = append(nodes, node)
    }

    return nodes, nil
}

// RetrieveChatsByVector retrieves chats from the associative memory based on vector similarity
func (am *AssociativeMemory) RetrieveChatsByVector(queryEmbedding []float32, k int) ([]ConceptNode, error) {
    am.mu.RLock()
    defer am.mu.RUnlock()

    var nodes []ConceptNode
    query := `
        SELECT cn.id, cn.node_count, cn.type_count, cn.type, cn.depth, cn.created, cn.expiration, cn.subject, cn.predicate, cn.object, cn.description, cn.embedding_id, cn.poignancy, cn.keywords, cn.filling
        FROM concept_nodes cn
        INNER JOIN embeddings e ON cn.embedding_id = e.id
        WHERE cn.type = 'chat'
        ORDER BY e.embedding <-> $1 ASC
        LIMIT $2
    `

    rows, err := am.db.Query(query, pq.Array(queryEmbedding), k)
    if err != nil {
        am.logger.Printf("Error retrieving chats by vector: %v", err)
        return nil, err
    }
    defer rows.Close()

    for rows.Next() {
        var node ConceptNode
        var expiration pq.NullTime
        err = rows.Scan(&node.ID, &node.NodeCount, &node.TypeCount, &node.Type, &node.Depth, &node.Created, &expiration, &node.Subject, &node.Predicate, &node.Object, &node.Description, &node.EmbeddingID, &node.Poignancy, pq.Array(&node.Keywords), pq.Array(&node.Filling))
        if err != nil {
            am.logger.Printf("Error scanning chat node: %v", err)
            return nil, err
        }
        if expiration.Valid {
            node.Expiration = &expiration.Time
        }
        nodes = append(nodes, node)
    }

    return nodes, nil
}

func (am *AssociativeMemory) initBM25() {
    var corpus []string
    rows, err := am.db.Query(`SELECT description FROM concept_nodes`)
    if err != nil {
        am.logger.Printf("Error retrieving corpus for BM25: %v", err)
        return
    }
    defer rows.Close()

    for rows.Next() {
        var description string
        err = rows.Scan(&description)
        if err != nil {
            am.logger.Printf("Error scanning description for BM25: %v", err)
            return
        }
        corpus = append(corpus, description)
    }

    am.bm25 = bm25.NewBM25(corpus)
}
