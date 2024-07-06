package concurrency

import (
	"container/heap"
	"llm-agent/internal/types"
	"sync"
)

// EventPriorityQueue is a priority queue for events
type EventPriorityQueue []*types.Event

// Len returns the length of the priority queue
func (pq EventPriorityQueue) Len() int { return len(pq) }

// Less compares the priority of two events
func (pq EventPriorityQueue) Less(i, j int) bool {
	return pq[i].Priority < pq[j].Priority
}

// Swap swaps the positions of two events in the priority queue
func (pq EventPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

// Push adds an event to the priority queue
func (pq *EventPriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*types.Event))
}

// Pop removes the event with the highest priority from the priority queue
func (pq *EventPriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	x := old[n-1]
	*pq = old[:n-1]
	return x
}

// ConcurrencyManager manages concurrency and thread pools
type ConcurrencyManager struct {
	heartbeatPool   *sync.WaitGroup
	eventPool       *sync.WaitGroup
	globalPool      *sync.WaitGroup
	eventQueue      EventPriorityQueue
	eventQueueMutex sync.Mutex
}

// NewConcurrencyManager creates a new instance of ConcurrencyManager
func NewConcurrencyManager(heartbeatPoolSize, eventPoolSize, globalPoolSize int) *ConcurrencyManager {
	return &ConcurrencyManager{
		heartbeatPool:   &sync.WaitGroup{},
		eventPool:       &sync.WaitGroup{},
		globalPool:      &sync.WaitGroup{},
		eventQueue:      make(EventPriorityQueue, 0),
	}
}

// EnqueueEvent adds an event to the event priority queue
func (cm *ConcurrencyManager) EnqueueEvent(event *types.Event) {
	cm.eventQueueMutex.Lock()
	defer cm.eventQueueMutex.Unlock()

	heap.Push(&cm.eventQueue, event)
}

// DequeueEvent removes the event with the highest priority from the event priority queue
func (cm *ConcurrencyManager) DequeueEvent() *types.Event {
	cm.eventQueueMutex.Lock()
	defer cm.eventQueueMutex.Unlock()

	if len(cm.eventQueue) == 0 {
		return nil
	}

	event := heap.Pop(&cm.eventQueue).(*types.Event)
	return event
}

// ProcessHeartbeat processes a heartbeat event
func (cm *ConcurrencyManager) ProcessHeartbeat(heartbeat func()) {
	cm.heartbeatPool.Add(1)
	go func() {
		defer cm.heartbeatPool.Done()
		heartbeat()
	}()
}

// ProcessEvent processes an event
func (cm *ConcurrencyManager) ProcessEvent(event *types.Event, handler func(*types.Event)) {
	if cm.eventPool.TryAdd(1) {
		go func() {
			defer cm.eventPool.Done()
			handler(event)
		}()
	} else {
		cm.EnqueueEvent(event)
	}
}

// ProcessGlobalTask processes a global task
func (cm *ConcurrencyManager) ProcessGlobalTask(task func()) {
	cm.globalPool.Add(1)
	go func() {
		defer cm.globalPool.Done()
		task()
	}()
}

// StartEventLoop starts the event loop for processing events from the priority queue
func (cm *ConcurrencyManager) StartEventLoop(handler func(*types.Event)) {
	go func() {
		for {
			event := cm.DequeueEvent()
			if event == nil {
				// No events in the queue, wait for a short time
				time.Sleep(100 * time.Millisecond)
				continue
			}

			cm.ProcessEvent(event, handler)
		}
	}()
}
