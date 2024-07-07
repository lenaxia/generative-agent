# Generative Agent v3.5

## System Design

### Description

You are building a modular and extensible LLM-based generative agent system capable of dynamic module loading, function calling, and integration with various services and tools. The system is designed to operate on heartbeats and events, allowing the agent to perform actions without external stimuli or respond to external stimuli, respectively.

The core of the system is the Service Locator, which acts as a central registry for managing and discovering available services. Services are defined as modules, with each module containing one or more services and any necessary assets (e.g., prompts, templates, configurations). The Service Locator interface provides descriptions of each service's capabilities and allows services to interact with LLM objects if needed.

The Service Locator separates services into core and extended capabilities. Core capabilities are always available, while extended capabilities are surfaced only when needed. Examples of services include interfaces for managing LLM memory through vector stores, managing knowledge bases for Retrieval-Augmented Generation (RAG), fetching weather data, interacting with the operating system or remote machines, and connecting to notification services like Slack or Discord.

The service interfaces also allow stateful services to raise events. For instance, Slack and Discord integration services can raise events when new messages arrive in monitored channels or when receiving direct messages. Another example of a stateful service is an SSH/RDP service that allows the LLM to maintain an open connection to a remote machine for interaction, including maintaining the connection state (e.g., purpose, tasks, progress).

The system includes a Generative Agent Core module responsible for orchestrating the agent's behavior, including planning, reasoning, execution, and state management. The agent can query the Service Locator for available services and their capabilities, decide which services to use, and call their functions as needed.

The agent operates on a cycle of heartbeats and events. Heartbeats trigger the agent to perform actions without external stimuli, while events allow the agent to respond to external stimuli. During each cycle, the agent makes a decision on what functions to call based on tools available through the service locator. It is allowed to iterate on the results of its function call until it decides its turn is complete and its ready to respond to the event or has no more work left to do on its current task. In the case of a heartbeat, an initial call to the LLM backend is made with a prompt based on the current state of the system, existing tasks and their status, and previous actions that have been taken recently. An event is then generated and we merge into the same workflow as the event pathway.

In order to determine what to do with an event, the LLM backend is presented with a list of core services, as well as any extended services that may be relevant. The LLM is prompted to decide on a function call pattern to process the event. This may be a simple "respond to the query" or could be a more complex task that relies on reasoning, refelction, and planning services.

For instance, if the event is a Slack raised event of "incoming message", the generative agent could just decide to call a "Respond to Slack" service and send a message directly back.

Or if the event is a heartbeat, the Generative Agent may decide to use a "check home assistant status" service that would result in calls to a "weather" service, "time" service, "calendar" service to understand the schedule and timing of people in and out of the house to determine how it wants to configure the HVAC, window blinds and lights.

Or finally if there is a capability that the generative agent cannot currently do, it could decide it wants to write a new module for itself, it would generate a plan and a design for what it wants, implement the code, then using SSH connect to a remote machine, and run and test the code it has written, once it is satisfied, it would then use the OS module to write the files to the modules folder for the module to be loaded into the store locator.

The system includes concurrency and thread management features to handle the load efficiently:

### Configurable Thread Pools

The system has separate thread pools for handling heartbeats, events, and global tasks. The thread pool sizes are configurable to allow for dynamic scaling based on the observed load. Event Queuing and Prioritization:

If the event thread pool is at capacity, events are queued in a priority queue. High-priority events can jump to the front of the queue. Before processing an event from the queue, the system makes an LLM call to verify if the event is still relevant. If not, the event is discarded. Global Thread Pool:

The system has a global thread pool, which is smaller than the heartbeat and event thread pools. If the global thread pool is exhausted, new tasks (both heartbeats and events) are queued and processed as resources become available. This concurrency and thread management approach allows the system to handle load spikes, prioritize important events, and gracefully degrade performance when resources are exhausted.

The system also includes a Module Manager responsible for dynamically loading, reloading, unloading, registering, and deregistering modules. Modules can be added or removed without restarting the core service, allowing the system's capabilities to grow or change over time. The Module Manager monitors a dedicated modules directory and ensures that all necessary files and assets are present before loading a module.

The Service Locator raises events when services are registered, deregistered, or when core services are registered or deregistered, allowing the Generative Agent Core to update its internal state accordingly.

Overall, this design promotes modularity, extensibility, and separation of concerns, enabling the LLM-based generative agent system to adapt and grow while maintaining a robust and flexible architecture.

### **Project Structure**

```
llm-agent/
├── cmd/
│   └── main.go
├── configs/
│   ├── config.go
│   └── config.yaml
├── internal/
│   ├── core/
│   │   ├── agent/
│   │   │   └── agent.go
│   │   ├── memory/
│   │   │   └── memory.go
│   │   ├── modulemanager/
│   │   │   └── modulemanager.go
│   │   ├── servicelocator/
│   │   │   └── servicelocator.go
│   │   └── types/
│   │       ├── event.go
│   │       ├── function.go
│   │       ├── plan.go
│   │       ├── result.go
│   │       └── state.go
│   ├── interfaces/
│   │   └── service.go
│   └── modules/
│       ├── vectorstore/
│       │   ├── main.go
│       │   └── vectorstore.go
│       ├── knowledgebase/
│       │   ├── main.go
│       │   └── knowledgebase.go
│       ├── weather/
│       │   ├── main.go
│       │   └── weather.go
│       ├── os/
│       │   ├── main.go
│       │   └── os.go
│       ├── notifications/
│       │   ├── main.go
│       │   ├── slack/
│       │   │   └── slack.go
│       │   └── discord/
│       │       └── discord.go
│       └── remoteconnection/
│           ├── main.go
│           ├── ssh/
│           │   └── ssh.go
│           └── rdp/
│               └── rdp.go
├── pkg/
│   ├── events/
│   │   └── events.go
│   └── utils/
│       └── utils.go
├── vendor/
└── go.mod
```
