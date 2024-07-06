package agent

import (
	"llm-agent/configs"
	"llm-agent/internal/core/concurrency"
	"llm-agent/internal/core/servicelocator"
	"llm-agent/internal/interfaces"
	"llm-agent/internal/types"
	"llm-agent/internal/core/eventmanager"
)

// GenerativeAgent is the core of the LLM-based generative agent system
type GenerativeAgent struct {
	serviceLocator     *servicelocator.ServiceLocator
	eventManager       *events.EventManager
	concurrencyManager *concurrency.ConcurrencyManager
	config             *configs.Config
	state              *types.State
}

// NewGenerativeAgent creates a new instance of GenerativeAgent
func NewGenerativeAgent(sl *servicelocator.ServiceLocator, em *events.EventManager, cfg *configs.Config) *GenerativeAgent {
	cm := concurrency.NewConcurrencyManager(cfg.HeartbeatPoolSize, cfg.EventPoolSize, cfg.GlobalPoolSize)
	return &GenerativeAgent{
		serviceLocator:     sl,
		eventManager:       em,
		concurrencyManager: cm,
		config:             cfg,
		state:              &types.State{},
	}
}

// Start starts the generative agent
func (a *GenerativeAgent) Start() {
	// Register event handlers
	a.eventManager.RegisterHandler(types.HeartbeatEvent, a.handleHeartbeatEvent)
	a.eventManager.RegisterHandler(types.ExternalEvent, a.handleExternalEvent)

	// Start the event loop
	a.eventManager.Start()

	// Start processing heartbeats
	a.startHeartbeats()

	// Start the event loop for processing events from the priority queue
	a.concurrencyManager.StartEventLoop(a.handleExternalEvent)
}

// startHeartbeats starts processing heartbeat events
func (a *GenerativeAgent) startHeartbeats() {
	heartbeatInterval := a.config.HeartbeatInterval
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			a.concurrencyManager.ProcessHeartbeat(a.handleHeartbeat)
		}
	}
}

// handleHeartbeat handles heartbeat events
func (a *GenerativeAgent) handleHeartbeat() {
	event := &types.Event{
		Type: types.HeartbeatEvent,
	}
	a.eventManager.Emit(event)
}

// handleHeartbeatEvent handles heartbeat events from the event manager
func (a *GenerativeAgent) handleHeartbeatEvent(event *types.Event) {
	// Get available services from the service locator
	services := a.serviceLocator.GetServices()

	// Call the LLM to generate a plan based on the current state and available services
	plan := a.generatePlan(a.state, services)

	// Execute the plan
	result := a.executePlan(plan)

	// Update the state based on the result
	a.state.Update(result)
}

// handleExternalEvent handles external events
func (a *GenerativeAgent) handleExternalEvent(event *types.Event) {
	// Check if the event is still relevant this should only occur when dequeing an event
    // If the event is being handled live, it is assumed to be still relevant 
	isRelevant, err := a.checkEventRelevance(event)
	if err != nil {
		// Handle error
		return
	}

	if !isRelevant {
		// Discard the event
		return
	}

	// Get available services from the service locator
	services := a.serviceLocator.GetServices()

	// Call the LLM to generate a plan based on the event and available services
	plan := a.generatePlan(event, services)

	// Execute the plan
	result := a.executePlan(plan)

	// Update the state based on the result
	a.state.Update(result)
}

// checkEventRelevance checks if an event is still relevant
func (a *GenerativeAgent) checkEventRelevance(event *types.Event) (bool, error) {
	// Get the LLM backend service
	llmService, err := a.serviceLocator.Get(reflect.TypeOf((*LLMBackend)(nil)).Elem(), "llm")
	if err != nil {
		return false, err
	}

	llmBackend, ok := llmService.(LLMBackend)
	if !ok {
		return false, fmt.Errorf("service is not an LLMBackend")
	}

	// Call the LLM backend to check if the event is still relevant
	isRelevant, err := llmBackend.CheckEventRelevance(event)
	if err != nil {
		return false, err
	}

	return isRelevant, nil
}

// generatePlan generates a plan based on the current state or event and available services
func (a *GenerativeAgent) generatePlan(input interface{}, services []interfaces.Service) *types.Plan {
	// Call the LLM to generate a plan
	// ...

	// Return the generated plan
	return &types.Plan{}
}

// executePlan executes a plan by calling the necessary service functions
func (a *GenerativeAgent) executePlan(plan *types.Plan) *types.Result {
	// Execute the plan by calling service functions
	// ...

	// Return the result
	return &types.Result{}
}
