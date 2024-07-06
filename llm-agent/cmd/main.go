package main

import (
	"fmt"
	"llm-agent/configs"
	"llm-agent/internal/core/agent"
	"llm-agent/internal/core/modulemanager"
	"llm-agent/internal/core/servicelocator"
	"llm-agent/internal/core/eventmanager"
)

func main() {
	// Load configuration
	cfg, err := configs.LoadConfig("configs/config.yaml")
	if err != nil {
		fmt.Println("Error loading configuration:", err)
		return
	}

	// Create service locator
	sl := servicelocator.NewServiceLocator()

	// Create event manager
	em := events.NewEventManager()

	// Create module manager
	mm, err := modulemanager.NewModuleManager(sl, cfg.ModulePath)
	if err != nil {
		fmt.Println("Error creating module manager:", err)
		return
	}

	// Create generative agent
	agent := agent.NewGenerativeAgent(sl, em, cfg)

	// Start the agent
	agent.Start()
}
