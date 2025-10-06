# API Reference

## Overview

This document provides comprehensive API documentation for the StrandsAgent Universal Agent system. The system provides a unified workflow management interface with role-based task delegation and external state management.

## Core Components

### WorkflowEngine

The `WorkflowEngine` is the primary orchestration component that manages workflow lifecycle, task scheduling, and Universal Agent integration.

#### Constructor

```python
WorkflowEngine(llm_factory: LLMFactory, message_bus: MessageBus, max_concurrent_tasks: int = 5)
```

**Parameters:**
- `llm_factory`: LLMFactory instance for creating StrandsAgent models
- `message_bus`: MessageBus instance for event communication
- `max_concurrent_tasks`: Maximum number of concurrent tasks (default: 5)

#### Methods

##### start_workflow(instruction: str) -> str

Creates and starts a new workflow from user instruction.

**Parameters:**
- `instruction`: User instruction describing the task to be performed

**Returns:**
- `str`: Unique workflow ID for tracking

**Example:**
```python
workflow_id = workflow_engine.start_workflow("Search for weather in Seattle and summarize the results")
```

##### pause_workflow(workflow_id: str) -> Dict

Pauses an active workflow and creates a comprehensive checkpoint.

**Parameters:**
- `workflow_id`: ID of the workflow to pause

**Returns:**
- `Dict`: Checkpoint data containing workflow state

**Example:**
```python
checkpoint = workflow_engine.pause_workflow("wf_12345")
```

##### resume_workflow(workflow_id: str, checkpoint: Dict) -> None

Resumes a paused workflow from checkpoint data.

**Parameters:**
- `workflow_id`: ID of the workflow to resume
- `checkpoint`: Checkpoint data from pause_workflow()

**Example:**
```python
workflow_engine.resume_workflow("wf_12345", checkpoint)
```

##### get_workflow_status(workflow_id: str) -> Dict

Gets the current status of a workflow.

**Parameters:**
- `workflow_id`: ID of the workflow to check

**Returns:**
- `Dict`: Workflow status information

**Example:**
```python
status = workflow_engine.get_workflow_status("wf_12345")
print(f"State: {status['state']}, Progress: {status['progress']}")
```

##### get_workflow_metrics() -> Dict

Gets comprehensive metrics for the WorkflowEngine.

**Returns:**
- `Dict`: Metrics including active workflows, queue size, and performance data

**Example:**
```python
metrics = workflow_engine.get_workflow_metrics()
print(f"Active workflows: {metrics['active_workflows']}")
```

##### get_universal_agent_status() -> Dict

Gets the status of the Universal Agent integration.

**Returns:**
- `Dict`: Universal Agent status and configuration

**Example:**
```python
ua_status = workflow_engine.get_universal_agent_status()
print(f"Framework: {ua_status['framework']}")
```

### Universal Agent

The `UniversalAgent` provides role-based task execution using StrandsAgent framework.

#### Constructor

```python
UniversalAgent(llm_factory: LLMFactory)
```

**Parameters:**
- `llm_factory`: LLMFactory instance for creating models

#### Methods

##### assume_role(role: str, llm_type: LLMType = LLMType.DEFAULT, context: TaskContext = None, tools: List[str] = None) -> Agent

Creates a role-specific agent instance.

**Parameters:**
- `role`: Role name (planning, search, weather, summarizer, slack)
- `llm_type`: LLM type for semantic model selection (WEAK, DEFAULT, STRONG)
- `context`: TaskContext for external state management
- `tools`: List of tool names to make available

**Returns:**
- `Agent`: StrandsAgent instance configured for the role

**Example:**
```python
agent = universal_agent.assume_role(
    role="planning",
    llm_type=LLMType.STRONG,
    context=task_context,
    tools=["create_task_plan", "analyze_dependencies"]
)
```

### LLMFactory

Enhanced factory for creating StrandsAgent models with semantic type mapping.

#### Constructor

```python
LLMFactory(configs: Dict[LLMType, List[BaseConfig]], framework: str = "strands")
```

**Parameters:**
- `configs`: Configuration mapping for different LLM types
- `framework`: Framework type ("strands" only)

#### Methods

##### create_strands_model(llm_type: LLMType, name: Optional[str] = None) -> Model

Creates a StrandsAgent model instance.

**Parameters:**
- `llm_type`: Semantic model type (WEAK, DEFAULT, STRONG)
- `name`: Optional specific model name

**Returns:**
- `Model`: StrandsAgent model instance

**Example:**
```python
model = llm_factory.create_strands_model(LLMType.STRONG)
```

##### create_universal_agent(llm_type: LLMType, role: str, tools: List = None) -> Agent

Creates a Universal Agent with specified configuration.

**Parameters:**
- `llm_type`: Semantic model type
- `role`: Agent role
- `tools`: Available tools

**Returns:**
- `Agent`: Configured Universal Agent instance

### TaskContext

External state management wrapper around TaskGraph.

#### Constructor

```python
TaskContext(task_graph: TaskGraph)
```

**Parameters:**
- `task_graph`: TaskGraph instance for DAG management

#### Methods

##### create_checkpoint() -> Dict

Creates a comprehensive checkpoint of current state.

**Returns:**
- `Dict`: Checkpoint data including task graph state, conversation history, and metadata

##### resume_from_checkpoint(checkpoint: Dict) -> None

Restores state from checkpoint data.

**Parameters:**
- `checkpoint`: Checkpoint data from create_checkpoint()

##### update_task_result(task_id: str, result: Any) -> None

Updates the result of a completed task.

**Parameters:**
- `task_id`: ID of the completed task
- `result`: Task execution result

##### get_conversation_history() -> List[Dict]

Gets the conversation history for the context.

**Returns:**
- `List[Dict]`: Conversation messages and metadata

## Tool Functions

The system includes several @tool decorated functions for specific capabilities:

### Planning Tools

Located in `llm_provider/planning_tools.py`:

- `create_task_plan(instruction: str, available_agents: List[str]) -> Dict`
- `validate_task_dependencies(task_plan: Dict) -> Dict`
- `optimize_task_execution_order(tasks: List[Dict]) -> List[Dict]`

### Search Tools

Located in `llm_provider/search_tools.py`:

- `web_search(query: str, num_results: int = 5) -> Dict`
- `search_with_filters(query: str, filters: Dict) -> Dict`

### Weather Tools

Located in `llm_provider/weather_tools.py`:

- `get_weather(location: str) -> Dict`
- `get_weather_forecast(location: str, days: int = 5) -> Dict`

### Summarizer Tools

Located in `llm_provider/summarizer_tools.py`:

- `summarize_text(text: str, max_length: int = 200) -> str`
- `extract_key_points(text: str, num_points: int = 5) -> List[str]`

### Slack Tools

Located in `llm_provider/slack_tools.py`:

- `send_slack_message(channel: str, message: str) -> Dict`
- `get_slack_channel_history(channel: str, limit: int = 10) -> List[Dict]`

## Configuration

### Main Configuration

The system uses a single `config.yaml` file with the following structure:

```yaml
# Logging configuration
logging:
  log_level: INFO
  log_file: logs/system.log

# LLM provider configurations
llm_providers:
  default:
    llm_class: DEFAULT
    provider_type: bedrock
    model_id: us.amazon.nova-pro-v1:0
    temperature: 0.3
  
  strong:
    llm_class: STRONG
    provider_type: bedrock
    model_id: us.amazon.nova-premier-v1:0
    temperature: 0.1
  
  weak:
    llm_class: WEAK
    provider_type: bedrock
    model_id: us.amazon.nova-lite-v1:0
    temperature: 0.5

# Universal Agent configuration
universal_agent:
  framework: strands
  max_concurrent_tasks: 5
  checkpoint_interval: 300
  
  # Role-based LLM optimization
  role_optimization:
    planning: STRONG      # Complex reasoning
    search: WEAK         # Simple queries
    weather: WEAK        # Simple lookups
    summarizer: DEFAULT  # Balanced processing
    slack: DEFAULT       # Conversational

# MCP server configuration
mcp:
  config_file: config/mcp_config.yaml
  
# Feature flags
features:
  universal_agent_enabled: true
  mcp_integration_enabled: true
  task_scheduling_enabled: true
  checkpoint_persistence: true

# Development settings
development:
  debug_mode: false
  test_mode: false
  mock_external_services: false
```

### Environment Variables

The system supports environment variable substitution using `${VAR:default}` syntax:

```yaml
llm_providers:
  default:
    model_id: ${BEDROCK_MODEL_ID:us.amazon.nova-pro-v1:0}
    temperature: ${TEMPERATURE:0.3}
```

## Error Handling

### Error Types

- `WorkflowError`: General workflow execution errors
- `TaskExecutionError`: Task-specific execution failures
- `ConfigurationError`: Configuration validation errors
- `UniversalAgentError`: Universal Agent creation or execution errors

### Error Response Format

```python
{
    "error": {
        "type": "TaskExecutionError",
        "message": "Task execution failed",
        "details": {
            "task_id": "task_123",
            "agent_role": "search",
            "retry_count": 2
        },
        "timestamp": "2025-10-02T06:50:00Z"
    }
}
```

## Event System

### Message Bus Events

The system publishes the following events:

#### TASK_STARTED
Published when a task begins execution.

```python
{
    "event_type": "TASK_STARTED",
    "workflow_id": "wf_12345",
    "task_id": "task_123",
    "agent_role": "planning",
    "timestamp": "2025-10-02T06:50:00Z"
}
```

#### TASK_COMPLETED
Published when a task completes successfully.

```python
{
    "event_type": "TASK_COMPLETED",
    "workflow_id": "wf_12345",
    "task_id": "task_123",
    "result": {...},
    "execution_time": 2.5,
    "timestamp": "2025-10-02T06:50:02Z"
}
```

#### TASK_FAILED
Published when a task fails.

```python
{
    "event_type": "TASK_FAILED",
    "workflow_id": "wf_12345",
    "task_id": "task_123",
    "error": "Task execution failed",
    "retry_count": 1,
    "timestamp": "2025-10-02T06:50:02Z"
}
```

#### WORKFLOW_PAUSED
Published when a workflow is paused.

```python
{
    "event_type": "WORKFLOW_PAUSED",
    "workflow_id": "wf_12345",
    "checkpoint_id": "cp_67890",
    "timestamp": "2025-10-02T06:50:05Z"
}
```

#### WORKFLOW_RESUMED
Published when a workflow is resumed.

```python
{
    "event_type": "WORKFLOW_RESUMED",
    "workflow_id": "wf_12345",
    "checkpoint_id": "cp_67890",
    "timestamp": "2025-10-02T06:50:10Z"
}
```

## Usage Examples

### Basic Workflow Execution

```python
from supervisor.workflow_engine import WorkflowEngine
from llm_provider.factory import LLMFactory, LLMType
from common.message_bus import MessageBus

# Initialize components
message_bus = MessageBus()
llm_factory = LLMFactory(configs, framework="strands")
workflow_engine = WorkflowEngine(llm_factory, message_bus)

# Start a workflow
workflow_id = workflow_engine.start_workflow(
    "Search for current weather in Seattle and create a summary report"
)

# Monitor progress
status = workflow_engine.get_workflow_status(workflow_id)
print(f"Workflow {workflow_id} is {status['state']}")
```

### Pause and Resume Workflow

```python
# Pause a running workflow
checkpoint = workflow_engine.pause_workflow(workflow_id)

# Save checkpoint for later use
import json
with open(f"checkpoint_{workflow_id}.json", "w") as f:
    json.dump(checkpoint, f)

# Resume workflow later
with open(f"checkpoint_{workflow_id}.json", "r") as f:
    checkpoint = json.load(f)

workflow_engine.resume_workflow(workflow_id, checkpoint)
```

### Custom Tool Integration

```python
from strands import tool

@tool
def custom_analysis_tool(data: str, analysis_type: str) -> Dict:
    """Custom analysis tool for specific domain tasks"""
    # Implementation here
    return {"analysis": "results", "confidence": 0.95}

# Register tool with Universal Agent
universal_agent.tool_registry.register_tool("custom_analysis", custom_analysis_tool)

# Use in workflow
agent = universal_agent.assume_role(
    role="analyst",
    llm_type=LLMType.STRONG,
    tools=["custom_analysis"]
)
```

### Event Subscription

```python
def handle_task_completion(event):
    print(f"Task {event['task_id']} completed in {event['execution_time']}s")

message_bus.subscribe("TASK_COMPLETED", handle_task_completion)
```

## Performance Considerations

### Model Selection Guidelines

- **WEAK**: Use for simple lookups, basic searches, and straightforward tasks
- **DEFAULT**: Use for balanced processing, conversational tasks, and general purpose
- **STRONG**: Use for complex reasoning, planning, and analysis tasks

### Concurrency Settings

- `max_concurrent_tasks`: Balance between throughput and resource usage
- Recommended values: 3-10 depending on system resources
- Monitor memory usage and API rate limits

### Checkpoint Strategy

- Checkpoints are created automatically during pause operations
- Manual checkpointing can be triggered for long-running workflows
- Checkpoint data includes full conversation history and task state

## Security Considerations

### API Access

- All API endpoints require proper authentication (when security is enabled)
- Role-based access control determines available operations
- Input validation is performed on all parameters

### Data Handling

- Conversation history may contain sensitive information
- Checkpoint data should be stored securely
- External tool integrations should use secure authentication

## Troubleshooting

### Common Issues

#### WorkflowEngine Creation Fails
- Check LLMFactory configuration
- Verify MessageBus initialization
- Ensure all required dependencies are installed

#### Task Execution Timeouts
- Increase timeout values in configuration
- Check external service availability
- Monitor system resource usage

#### Universal Agent Role Errors
- Verify role name is supported (planning, search, weather, summarizer, slack)
- Check tool availability for the role
- Ensure LLMType is appropriate for the task complexity

### Debug Mode

Enable debug mode in configuration for detailed logging:

```yaml
development:
  debug_mode: true
```

### Health Checks

The system provides health check endpoints:

```python
# Check overall system health
health = workflow_engine.get_system_health()

# Check specific component health
ua_health = workflow_engine.get_universal_agent_health()
mcp_health = workflow_engine.get_mcp_health()
```

## Migration from Legacy System

### RequestManager Compatibility

The WorkflowEngine maintains backward compatibility with RequestManager API:

```python
# Legacy RequestManager usage still works
request_id = workflow_engine.handle_request(instruction)
status = workflow_engine.get_request_status(request_id)
```

### Agent ID Mapping

Legacy agent IDs are automatically mapped to roles:

- `planning_agent` → `planning`
- `search_agent` → `search`
- `weather_agent` → `weather`
- `summarizer_agent` → `summarizer`
- `slack_agent` → `slack`

### Configuration Migration

Legacy agent configurations are automatically converted:

```yaml
# Legacy format (still supported)
agents:
  planning_agent:
    enabled: true
    model_type: strong

# New format (recommended)
universal_agent:
  role_optimization:
    planning: STRONG