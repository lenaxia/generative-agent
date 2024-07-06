package eventmanager

import (
	"llm-agent/internal/types"
	"sync"
)

// EventManager manages the event system
type EventManager struct {
	handlers map[types.EventType][]EventHandler
	queue    chan *types.Event
	wg       sync.WaitGroup
	quit     chan struct{}
}

// EventHandler is a function that handles an event
type EventHandler func(*types.Event)

// NewEventManager creates a new instance of EventManager
func NewEventManager() *EventManager {
	return &EventManager{
		handlers: make(map[types.EventType][]EventHandler),
		queue:    make(chan *types.Event, 100),
		quit:     make(chan struct{}),
	}
}

// RegisterHandler registers an event handler
func (em *EventManager) RegisterHandler(eventType types.EventType, handler EventHandler) {
	em.handlers[eventType] = append(em.handlers[eventType], handler)
}

// Emit emits an event
func (em *EventManager) Emit(event *types.Event) {
	em.queue <- event
}

// Start starts the event loop
func (em *EventManager) Start() {
	em.wg.Add(1)
	go em.eventLoop()
}

// Stop stops the event loop
func (em *EventManager) Stop() {
	close(em.quit)
	em.wg.Wait()
}

// eventLoop is the main event loop
func (em *EventManager) eventLoop() {
	defer em.wg.Done()

	for {
		select {
		case event := <-em.queue:
			em.handleEvent(event)
		case <-em.quit:
			return
		}
	}
}

// handleEvent handles an event by calling registered handlers
func (em *EventManager) handleEvent(event *types.Event) {
	for _, handler := range em.handlers[event.Type] {
		handler(event)
	}
}
