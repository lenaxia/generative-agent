package types

// CoreMemory is the central hub for managing the agent's memory components
type CoreMemory struct {
    serviceLocator *servicelocator.ServiceLocator
    storage        MemoryStorage
    cache          MemoryCache
    associative    *AssociativeMemory
    logger         *log.Logger
    mu             sync.RWMutex
}

// SearchResults represents the results of a memory search
type SearchResults struct {
    PlaintextResults []ConceptNode
    VectorResults    []ConceptNode
}

///////////////////////////
// Associative Memory

// Filter represents a filter for retrieving concept nodes
type FilterMemory struct {
    NodeType      string
    Subject       string
    Predicate     string
    Object        string
    Keywords      []string
    PlaintextQuery string
    StartTime     *time.Time
    EndTime       *time.Time
    EmbeddingIDs  []string
}

// ConceptNode represents a node in the associative memory
type ConceptNode struct {
    ID          string         `json:"id"`
    NodeCount   int            `json:"node_count"`
    TypeCount   int            `json:"type_count"`
    Type        string         `json:"type"`
    Depth       int            `json:"depth"`
    Created     time.Time      `json:"created"`
    Expiration  *time.Time     `json:"expiration,omitempty"`
    Subject     string         `json:"subject"`
    Predicate   string         `json:"predicate"`
    Object      string         `json:"object"`
    Description string         `json:"description"`
    EmbeddingID string         `json:"embedding_id"`
    Poignancy   float32        `json:"poignancy"`
    Keywords    pq.StringArray `json:"keywords"`
    Filling     []string       `json:"filling,omitempty"`
}


//////////////////////////////////
// Scratch Memory

// Filter represents a filter for retrieving scratches
type FilterScratch struct {
        Name          string
        FirstName     string
        LastName      string
        Age           int
        Innate        string
        Learned       string
        Currently     string
        Lifestyle     string
        LivingArea    string
        DailyPlanReq  string
        ActDescription string
        ActEvent      [3]string
        ActObjEvent   [3]string
        PlaintextQuery string
        StartTime     time.Time
        EndTime       time.Time
        EmbeddingIDs  []string
}


///////////////////////////////
// Spatial Memory

// FilterLocation represents a filter for retrieving locations
type FilterLocation struct {
    World         string
    Sector        string
    Arena         string
    Objects       []string
    PlaintextQuery string
}

// Location represents a location in the memory tree
type Location struct {
        World    string
        Sector   string
        Arena    string
        GpsCoords string
        Address  string
        Objects  []string
        Embedding []float32
}
