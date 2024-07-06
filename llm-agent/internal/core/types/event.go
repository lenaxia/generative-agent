package types

// EventType represents the type of an event
type EventType int

const (
	HeartbeatEvent EventType = iota
	ExternalEvent
)

// Event represents an event in the system
type Event struct {
	Type     EventType
	Data     interface{}
	Priority int
}
