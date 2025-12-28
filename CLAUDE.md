# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Development Commands

### Setup

```bash
# Full development setup with Docker Redis
make docker-setup

# Install development dependencies only
make install-dev

# Setup pre-commit hooks
make setup-pre-commit
```

### Testing

```bash
# Run all tests with coverage
make test

# Run specific test types
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-llm          # LLM provider tests only

# Run single test file
python -m pytest tests/unit/test_specific.py -v

# Run single test function
python -m pytest tests/unit/test_file.py::test_function_name -v

# Run with Docker Redis
make docker-test
```

### Code Quality

```bash
# Format code (black, isort)
make format

# Run all linters (flake8, mypy, pylint, bandit, yamllint)
make lint

# Auto-fix safe issues (unused imports, formatting)
make auto-fix-safe

# Auto-fix all issues (includes unused variables)
make auto-fix

# Run everything (format check, lint, all tests)
make check-all
```

### Running the System

```bash
# Interactive mode
python cli.py

# Single workflow execution
python cli.py --workflow "Set a timer for 10 minutes"

# Check system status
python cli.py --status

# Use custom configuration
python cli.py --config production.yaml
```

### Docker Commands

```bash
# Start Docker Redis
make docker-start

# Stop Docker containers
make docker-stop

# View logs
make docker-logs

# Connect to Redis CLI
make redis-cli

# Check Docker environment
make docker-check
```

---

## High-Level Architecture

### System Overview

**Universal Agent System**: A single AI agent that dynamically assumes specialized roles through fast-reply routing. Built on Strands Agent Framework with event-driven architecture and intent-based processing.

**Core Concept**: One agent + dynamic role specialization + intelligent routing + unified workflow management.

### Architecture Patterns (Hybrid Multi-Phase)

The system uses **three coexisting patterns**:

1. **Legacy Single-File Roles** (System Services)

   - `roles/core_router.py` - Request routing with confidence-based classification
   - `roles/core_conversation.py` - Fast-reply conversation handling
   - `roles/core_summarizer.py` - Fast-reply summarization (legacy, deprecated)
   - `roles/core_planning.py` - Meta-planning and agent configuration

2. **Phase 3 Domain Roles** (Event-Driven, Fast-Reply)

   - `roles/timer/` - Timer management with expiry detection
   - `roles/calendar/` - Calendar operations
   - `roles/weather/` - Weather queries
   - `roles/smart_home/` - Home Assistant integration
   - `roles/search/` - Web and news search using Tavily API

3. **Phase 4 Meta-Planning** (Dynamic Agent Creation)

   - Intelligent tool selection from registry
   - Runtime agent configuration
   - **Structured execution plans** with step-by-step guidance
   - **Dynamic replanning** on failures
   - Autonomous execution with 10-15 iterations

4. **Infrastructure Tools**
   - `tools/core/` - System infrastructure (memory, notification, summarization, planning)
   - `tools/custom/` - User-extensible tools (add custom tools here)

### Phase 3 Domain Role Structure

```
roles/{domain}/
├── role.py         # Role class with configuration
├── handlers.py     # Event handlers & intent processors
└── tools.py        # Tool implementations
```

**Role Class API:**

```python
class DomainRole:
    REQUIRED_TOOLS = ["domain.tool1", "domain.tool2"]

    async def initialize()              # Load tools from registry
    def get_system_prompt() -> str      # Role-specific prompt
    def get_llm_type() -> LLMType       # LLM tier (WEAK/DEFAULT/STRONG)
    def get_tools() -> list             # Loaded tool instances
    def get_role_config() -> dict       # Metadata & fast_reply flag
    def get_event_handlers() -> dict    # Event type -> handler function
    def get_intent_handlers() -> dict   # Intent class -> processor function
```

### Key Components

**Supervisor** (`supervisor/supervisor.py`)

- Top-level orchestrator
- Initializes all system components
- Manages communication channels (Slack, Discord, MQTT)
- Coordinates workflows through WorkflowEngine

**WorkflowEngine** (`supervisor/workflow_engine.py`)

- Executes task graphs and workflows
- Routes requests to appropriate roles
- Manages UniversalAgent lifecycle
- Handles result aggregation

**UniversalAgent** (`llm_provider/universal_agent.py`)

- Single agent interface with dynamic role assumption
- Manages agent pooling for efficiency
- Integrates with lifecycle hooks (pre-processors, post-processors)
- Executes with role-specific tools and prompts

**RoleRegistry** (`llm_provider/role_registry.py`)

- Discovers and loads all roles (legacy + domain)
- Manages fast-reply role identification
- Registers event handlers with MessageBus
- Registers intent handlers with IntentProcessor

**ToolRegistry** (`llm_provider/tool_registry.py`)

- Centralized tool discovery and loading
- Loads from `roles/{domain}/tools.py` for domain tools
- Loads from `tools/core/*.py` for infrastructure tools
- Provides tools to roles via `get_tools(REQUIRED_TOOLS)`

**MessageBus** (`common/message_bus.py`)

- Pub/sub event system
- Routes events to registered handlers
- Enables decoupled role communication

**IntentProcessor** (`common/intent_processor.py`)

- Processes Intent objects returned by event handlers
- Executes intent-specific logic (NotificationIntent, AuditIntent, etc.)
- LLM-safe architecture (scheduled execution, no direct asyncio)

### Memory Architecture

**Dual-Layer System:**

- **Layer 1**: Realtime Log (last N messages, 24h TTL) - Fast retrieval
- **Layer 2**: Assessed Memories (LLM-scored importance, graduated TTL) - Long-term storage

**UniversalMemoryProvider** (`common/memory_providers.py`)

- Redis-backed storage
- Automatic importance assessment via LLM
- Memory types: conversation, event, plan, preference, fact
- Graduated TTL based on importance score

### Fast-Reply Routing (Phase 3)

**How it Works:**

1. Router analyzes request with LLM
2. Returns role name + confidence score (0-1)
3. If confidence ≥ 0.95 → Fast-reply execution (~600ms)
4. If confidence < 0.70 → Phase 4 meta-planning (8-16s)
5. Fast-reply roles execute immediately with specialized tools

**Fast-Reply Roles (6 total):**

- Domain: timer, calendar, weather, smart_home, search
- Legacy: conversation

**Configuration:**

```python
def get_role_config(self) -> dict:
    return {
        "fast_reply": True,    # Enable fast-reply
        "llm_type": "WEAK",    # Or "DEFAULT", "STRONG"
    }
```

**Performance**: ~600ms for simple single-domain requests

---

### Phase 4 Meta-Planning with Execution Plans

**When to Use:**

- Multi-domain coordination (weather + timer)
- Complex workflows (search + summarize + notify)
- Multiple operations (check weather AND set timer AND add to calendar)
- Confidence < 0.70 from router

**How it Works:**

1. **Tool Selection**: LLM analyzes request, selects 2-3 tools from registry
2. **Execution Planning**: Creates structured ExecutionPlan with steps ⭐
3. **Agent Creation**: RuntimeAgentFactory creates custom agent
4. **Guided Execution**: Agent follows plan, can call replan() if needed ⭐
5. **Intent Processing**: Collects and processes intents
6. **Result**: Returns synthesized response

**Key Components:**

- `plan_and_configure_agent()` - Meta-planning with tool selection
- `create_execution_plan()` - Structured plan creation ⭐
- `replan()` - Dynamic plan revision on failures ⭐
- `SimplifiedWorkflowEngine` - Execution orchestration
- `RuntimeAgentFactory` - Custom agent creation

**Execution Plans:**

```python
ExecutionPlan(
    plan_id="plan_abc123",
    steps=[
        ExecutionStep(step_number=1, tool_name="weather.get_current_weather"),
        ExecutionStep(step_number=2, tool_name="timer.set_timer", depends_on=[1]),
    ],
    reasoning="Sequential execution",
    status=PlanStatus.PENDING
)
```

**Benefits:**

- ✅ Structured guidance keeps agents on track
- ✅ Dynamic replanning on failures
- ✅ Type-safe with Pydantic validation
- ✅ Observability with plan IDs
- ✅ No code changes for new workflows

**Performance**: 8-16s for complex multi-domain workflows

**See**: [docs/ROUTING_ARCHITECTURE.md](docs/ROUTING_ARCHITECTURE.md), [docs/EXECUTION_PLANNING_GUIDE.md](docs/EXECUTION_PLANNING_GUIDE.md)

### Event-Driven Flow

```
User Request
    ↓
Supervisor receives via CommunicationManager
    ↓
WorkflowEngine.process_request()
    ↓
Router analyzes → Returns (role, confidence)
    ↓
    ├─────────────────────────┬─────────────────────────┐
    ↓                         ↓                         ↓
[Confidence ≥ 0.95]    [0.70 ≤ Conf < 0.95]    [Confidence < 0.70]
Fast-Reply Path        Context-dependent        Meta-Planning Path
    ↓                         ↓                         ↓
UniversalAgent            May use either           plan_and_configure_agent()
    ↓                      pathway                      ↓
assume_role(role)                            LLM selects tools from registry
    ↓                                                   ↓
RoleRegistry.get_domain_role(role)          create_execution_plan() ⭐
    ↓                                                   ↓
Extract: tools, system_prompt, llm_type      Add replan tool to toolset ⭐
    ↓                                                   ↓
Execute with lifecycle hooks              RuntimeAgentFactory.create_agent()
    ↓                                                   ↓
Collect intents                           Agent executes with plan guidance ⭐
    ↓                                                   ↓
IntentProcessor processes intents         Can call replan() if steps fail ⭐
    ↓                                                   ↓
Return result                              Collect and process intents
                                                       ↓
                                            Return synthesized result
```

### Intent-Based Processing

**Pure Function Pattern (LLM-Safe):**

```python
# Event handler (pure function)
def handle_event(event_data: Any, context: LLMSafeEventContext) -> list[Intent]:
    """Process event, return intents for execution."""
    return [
        NotificationIntent(message="Timer expired", channel=context.channel_id),
        AuditIntent(action="timer_expiry", details={...})
    ]

# Intent processor (async, executed later)
async def process_notification_intent(intent: NotificationIntent):
    """Execute the actual notification."""
    await communication_manager.send(intent.message, intent.channel)
```

**Key Intents:**

- `NotificationIntent` - Send user notifications
- `AuditIntent` - Log actions for auditing
- `WorkflowExecutionIntent` - Trigger sub-workflows
- `TimerExpiryIntent`, `CalendarIntent`, etc. - Domain-specific

---

## LLM-Friendly Code Principles

This codebase follows **LLM-friendly patterns** for easier understanding and modification:

### 1. Locality of Behavior

Related code is physically close together. Avoid deep inheritance hierarchies.

### 2. Explicit Over Implicit

Use explicit imports, clear function names, obvious behavior. Avoid magic methods.

### 3. Flat Over Nested

Minimize inheritance and nesting. Prefer composition and explicit function calls.

### 4. Self-Documenting

Use type hints, descriptive names, comprehensive docstrings. Code explains itself.

### 5. Minimal Abstraction

Don't abstract too early. One level of abstraction is usually enough.

**Read more:** `docs/64_LLM_FRIENDLY_REFACTORING_PRINCIPLES.md`

---

## Collaboration Guidelines

### Always Ask About:

1. **Implementation Approach** - When multiple valid options exist

   - "Should this be a domain role or a tool?"
   - "Fast-reply role or multi-step workflow?"
   - "Event handler or direct implementation?"

2. **Architecture Decisions** - When changes affect system design

   - "Should I create a new role or extend existing?"
   - "Tools in `tools/core/` (infrastructure) or `tools/custom/` (user)?"
   - "Should this integrate with MessageBus events?"

3. **Breaking Changes** - Always confirm before:

   - Deleting files or code
   - Changing public APIs
   - Modifying role interfaces
   - Refactoring large sections

4. **Scope** - When requests could be interpreted broadly or narrowly
   - Clarify what "improve", "fix", "add" means
   - Define boundaries before proceeding
   - Ask about edge cases and requirements

### Present Options, Not Decisions

When multiple approaches are valid, present options with pros/cons and let the user choose.

**Example:**

```
"I see two ways to implement this:

Option A: Add to existing timer role
- Pros: Reuses existing infrastructure, simple
- Cons: Increases role complexity

Option B: Create new domain role
- Pros: Clean separation, follows Phase 3 pattern
- Cons: More files, additional setup

Which fits your needs better?"
```

### State Your Understanding

Before implementing significant features, restate your understanding:

```
"I understand you want to:
1. [Goal 1]
2. [Goal 2]

My plan:
- [Step 1]
- [Step 2]

Does this match your expectations?"
```

### Acknowledge Uncertainty

If unsure, say so explicitly:

- "I'm not certain about the best approach here. Let me investigate."
- "This could work, but I'd like to check [X] first."
- "I see potential issues with [Y]. Should I explore alternatives?"

---

## Key Architectural Decisions

### When to Create a Domain Role vs. Tool

**Create Domain Role When:**

- Users directly request this functionality ("set a timer", "check weather")
- Requires specialized system prompts
- Needs event handling or intent processing
- Should be selectable by router
- Benefits from fast-reply execution

**Create Tool When:**

- Used BY other roles, not invoked directly
- Infrastructure/utility function
- No specialized prompting needed
- Service layer, not agent layer

**Tools in `tools/core/`**: System infrastructure - memory, notification, **summarization**, **planning**

**Tools in `tools/custom/`**: User-extensible tools - Add custom capabilities here

### Fast-Reply vs. Meta-Planning

**Fast-Reply Roles (Phase 3):**

- Single-purpose operations
- Single domain only
- Quick information retrieval (~600ms)
- Set `fast_reply: True` in `get_role_config()`
- Examples: "set timer", "check weather", "search for X"

**Meta-Planning Workflows (Phase 4):**

- Complex, multi-domain operations
- Requires tool coordination
- LLM-driven tool selection
- Structured execution plans with replanning
- Performance: 8-16s
- Examples: "check weather AND set timer", "search + summarize"

---

## Testing Patterns

### Domain Role Testing

```python
# Test role initialization
async def test_role_initialization():
    role = TimerRole(tool_registry, llm_factory)
    await role.initialize()
    assert len(role.tools) == 3

# Test event handlers (pure functions)
def test_event_handler():
    context = LLMSafeEventContext(user_id="test", channel_id="channel")
    intents = handle_timer_expiry(event_data, context)
    assert isinstance(intents[0], TimerExpiryIntent)

# Test intent processors
async def test_intent_processor():
    intent = NotificationIntent(message="test", channel="test")
    await process_notification_intent(intent)
    # Verify notification sent
```

### Testing with Mocks

```python
# Mock providers for testing
class MockProvider:
    pass

providers = type('Providers', (), {
    'memory': MockProvider(),
    'communication': MockProvider(),
})()

await tool_registry.initialize(config={}, providers=providers)
```

---

## Common Patterns

### Adding a New Domain Role

1. Create directory: `roles/{domain}/`
2. Create `role.py` with DomainRole class
3. Create `handlers.py` with event handlers and intent processors
4. Create `tools.py` with tool implementations
5. Add to ToolRegistry domain_modules list
6. Register in RoleRegistry (automatic discovery)

### Adding Event Handlers

```python
# In handlers.py
def handle_my_event(event_data: Any, context: LLMSafeEventContext) -> list[Intent]:
    """LLM-SAFE: Pure function."""
    return [MyIntent(...)]

# In role.py
def get_event_handlers(self):
    from roles.my_domain.handlers import handle_my_event
    return {"MY_EVENT_TYPE": handle_my_event}
```

### Adding Intent Processors

```python
# In handlers.py
@dataclass
class MyIntent(Intent):
    field: str
    def validate(self) -> bool:
        return bool(self.field)

async def process_my_intent(intent: MyIntent):
    """Execute async operations."""
    await do_something(intent.field)

# In role.py
def get_intent_handlers(self):
    from roles.my_domain.handlers import MyIntent, process_my_intent
    return {MyIntent: process_my_intent}
```

---

## Important Files to Understand

**Entry Point:**

- `cli.py` - Command-line interface

**Core Orchestration:**

- `supervisor/supervisor.py` - Top-level coordinator
- `supervisor/workflow_engine.py` - Workflow execution engine (Phase 3 + Phase 4)
- `supervisor/simplified_workflow_engine.py` - Phase 4 meta-planning execution

**Agent System:**

- `llm_provider/universal_agent.py` - Single agent with role assumption
- `llm_provider/role_registry.py` - Role discovery and management
- `llm_provider/tool_registry.py` - Tool discovery and loading
- `llm_provider/runtime_agent_factory.py` - Dynamic agent creation (Phase 4)
- `llm_provider/factory.py` - LLM provider factory

**Planning System (Phase 4):**

- `roles/core_planning.py` - Meta-planning and agent configuration
- `roles/planning/tools.py` - Execution planning tools
- `common/planning_types.py` - Type-safe planning data structures
- `common/intent_collector.py` - LLM-safe intent collection

**Event System:**

- `common/message_bus.py` - Event pub/sub system
- `common/intent_processor.py` - Intent execution system
- `common/intents.py` - Core intent definitions

**Memory:**

- `common/memory_providers.py` - Dual-layer memory system

**Communication:**

- `communication/communication_manager.py` - Multi-platform messaging

**Infrastructure Tools:**

- `tools/core/memory.py` - Memory storage/retrieval
- `tools/core/notification.py` - Notification dispatch
- `tools/core/summarization.py` - Summarization and synthesis
- `tools/core/` - DO NOT MODIFY without understanding system impact

---

## Documentation to Read

**Architecture:**

- `docs/ROUTING_ARCHITECTURE.md` - **Phase 3 vs Phase 4 routing logic** ⭐
- `docs/EXECUTION_PLANNING_GUIDE.md` - **Execution plans and replanning** ⭐
- `docs/64_LLM_FRIENDLY_REFACTORING_PRINCIPLES.md` - Code patterns
- `docs/19_DYNAMIC_EVENT_DRIVEN_ROLE_ARCHITECTURE_DESIGN.md` - Event system
- `docs/24_UNIFIED_INTENT_PROCESSING_ARCHITECTURE.md` - Intent architecture
- `REFACTORING_STATUS_REVIEW.md` - Architecture status (Dec 22)

**Phase Documentation:**

- `PHASE3_LIFECYCLE_COMPLETE.md` - Domain role lifecycle
- `PHASE4_FINAL_STATUS.md` - **Phase 4 meta-planning complete** ⭐
- `PHASE4_FINAL_VALIDATION.md` - **Phase 4 validation results** ⭐
- `TOOLS_REORGANIZATION_PLAN.md` - Tools structure

**Type-Safe Planning:**

- `common/planning_types.py` - **ExecutionPlan, ExecutionStep types** ⭐
- `tests/unit/test_planning_types.py` - **27 type validation tests** ⭐
- `tests/unit/test_planning_tools.py` - **17 tool tests** ⭐

---

**Last Updated:** 2025-12-27
**Codebase Version:** Phase 4 Meta-Planning with Structured Execution Plans
