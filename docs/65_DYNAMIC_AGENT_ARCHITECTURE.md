# Dynamic Agent Architecture with Central Tool Registry

**Document ID:** 65
**Created:** 2025-01-20
**Status:** DESIGN
**Priority:** High
**Context:** Evolution from DAG-based workflows to dynamic agent creation with runtime tool selection

## Executive Summary

This document describes a major architectural evolution moving from rigid DAG-based workflows with predefined roles to flexible dynamic agent creation with runtime tool selection.

**Key Changes:**
1. Central Tool Registry - All tools discoverable in one place
2. Dynamic Agent Creation - Runtime agent configuration based on request needs
3. Meta-Planning - Planning agent selects tools and creates custom agents
4. Domain-Based Organization - Tools and roles organized by domain
5. Hybrid Intent System - Query tools execute directly, action tools use intents
6. Simplified Workflow - No DAG, just Plan → Create → Execute

**Goals:**
- Increase flexibility (handle novel requests)
- Reduce complexity (remove DAG engine)
- Improve maintainability (clear domain ownership)
- Preserve speed (fast-path routing)
- Maintain safety (intent system)

## Current vs New Architecture

### Current Flow
```
Request → Router → Planning → Creates DAG with predefined roles
                              ↓
                    WorkflowEngine executes: Role A → Role B → Role C
                              ↓
                    Task result sharing, progressive summarization
                              ↓
                    Response
```

**Limitations:**
- ❌ Inflexible (can't handle unexpected combinations)
- ❌ Complex (DAG execution, result sharing, checkpointing)
- ❌ Tool duplication across roles
- ❌ Maintenance burden

### New Flow
```
Request → Router → Confidence check
                      ↓
         High (≥95%)  ↓  Low (<70%)
                      ↓
    Predefined Role  ↓  Meta-Planning Agent
    (Fast, ~600ms)   ↓
                     ↓
    Analyzes request + ALL tools + context
                     ↓
    Outputs: plan + tool selection + agent config
                     ↓
    Runtime Agent Factory creates Strands agent
                     ↓
    Agent runs autonomously with selected tools
                     ↓
    Collect intents → Process → Return
```

**Benefits:**
- ✅ Flexible (any tool combination)
- ✅ Simple (no DAG)
- ✅ Leverages Strands Agent framework
- ✅ Fast path preserved
- ✅ Easy to extend

## Central Tool Registry

### Purpose
Single source of truth for ALL system capabilities.

### Implementation
```python
# llm_provider/tool_registry.py

class ToolRegistry:
    """Central registry for all tools"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    async def initialize(self, config, providers):
        """Load tools from domain modules"""
        domain_modules = [
            ("weather", "roles.weather.tools", providers.weather),
            ("calendar", "roles.calendar.tools", providers.calendar),
            ("timer", "roles.timer.tools", providers.redis),
            ("smart_home", "roles.smart_home.tools", providers.home_assistant),
            ("memory", "roles.memory.tools", providers.memory),
            ("search", "roles.search.tools", providers.search),
        ]

        for category, module_path, provider in domain_modules:
            tools = await self._load_tools_from_module(module_path, provider)
            self._register_tools(category, tools)

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool by fully qualified name (e.g., 'weather.get_forecast')"""
        return self._tools.get(tool_name)

    def get_tools(self, tool_names: List[str]) -> List[Tool]:
        """Get multiple tools by name"""
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self._tools.values())

    def format_for_llm(self) -> str:
        """Format all tools for LLM consumption (meta-planning)"""
        output = [f"AVAILABLE TOOLS ({len(self._tools)} total)\n"]

        for category in sorted(self._categories.keys()):
            tools = self.get_tools_by_category(category)
            output.append(f"\n{category.upper()} ({len(tools)} tools):")

            for tool in tools:
                output.append(f"  • {category}.{tool.name}: {tool.description}")

        return "\n".join(output)
```

**Tool Naming:** `<category>.<tool_name>` (e.g., `weather.get_forecast`, `calendar.create_event`)

## Domain Organization

### Directory Structure
```
roles/
├── weather/              # Domain with predefined role
│   ├── __init__.py
│   ├── tools.py         # Weather tools
│   └── role.py          # WeatherRole (fast path)
├── calendar/            # Domain with predefined role
│   ├── __init__.py
│   ├── tools.py
│   └── role.py
├── timer/               # Domain with predefined role
│   ├── __init__.py
│   ├── tools.py
│   └── role.py
├── smart_home/          # Domain with predefined role
│   ├── __init__.py
│   ├── tools.py
│   └── role.py
├── memory/              # Utility domain (no role)
│   ├── __init__.py
│   └── tools.py
├── search/              # Utility domain (no role)
│   ├── __init__.py
│   └── tools.py
├── notification/        # Utility domain (no role)
│   ├── __init__.py
│   └── tools.py
├── planning/            # Utility domain (executive planning)
│   ├── __init__.py
│   └── tools.py
├── shared/              # Cross-cutting utilities
│   └── lifecycle_helpers.py
├── core_router.py              # Multi-domain role
├── core_planning.py            # Multi-domain role (meta-planner)
├── core_conversation.py        # Multi-domain role
└── core_proactive_agent.py     # Multi-domain role
```

**Principles:**
- **Domain with Role:** Contains tools.py + role.py (e.g., weather, calendar)
- **Utility Domain:** Only tools.py, used by multiple roles (e.g., memory, search)
- **Multi-Domain Roles:** Top-level files using tools from multiple domains

### Tool Module Pattern
```python
# roles/weather/tools.py

from strands import Tool

def create_weather_tools(weather_provider) -> List[Tool]:
    """Create all weather-related tools"""

    async def get_current_weather(location: str) -> str:
        """Get current weather for a location."""
        result = await weather_provider.get_current(location)
        return format_weather(result)

    async def get_forecast(location: str, days: int = 3) -> str:
        """Get weather forecast."""
        result = await weather_provider.get_forecast(location, days)
        return format_forecast(result)

    # Helper function (not a tool)
    def format_weather(data) -> str:
        return f"Temperature: {data.temp}°F, Conditions: {data.conditions}"

    return [
        Tool.from_function(get_current_weather, category="weather"),
        Tool.from_function(get_forecast, category="weather"),
    ]
```

### Role Pattern
```python
# roles/weather/role.py

class WeatherRole:
    """Predefined role for fast weather queries"""

    # Declaration
    NAME = "weather"
    DESCRIPTION = "Provides weather information and forecasts"

    REQUIRED_TOOLS = [
        "weather.get_current_weather",
        "weather.get_forecast",
        "memory.save",  # Can use tools from other domains
    ]

    SYSTEM_PROMPT = """You are a weather information assistant..."""
    MODEL = "haiku"  # Fast model
    MAX_ITERATIONS = 5

    # Initialization
    def __init__(self, tool_registry, llm_factory):
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.tools = None

    async def initialize(self):
        """Load tools from registry"""
        self.tools = self.tool_registry.get_tools(self.REQUIRED_TOOLS)

    # Execution
    async def execute(self, request: str, context: TaskContext) -> RoleResult:
        """Fast execution - no meta-planning overhead"""
        intent_collector = IntentCollector()
        set_current_collector(intent_collector)

        try:
            agent = Agent(
                model=self.llm_factory.get_model(self.MODEL),
                system_prompt=self._build_system_prompt(context),
                tools=self.tools,
                max_iterations=self.MAX_ITERATIONS,
            )

            result = await agent.run(request)
            intents = intent_collector.get_intents()
            await self.intent_processor.process_intents(intents)

            return RoleResult(
                response=result.final_output,
                intents=intents,
                metadata={"role": self.NAME, "iterations": result.iteration_count}
            )
        finally:
            clear_current_collector()
```

**Key Points:**
- Roles **declare** tools (don't define them)
- Tools **loaded** from central registry
- Roles are **optimization patterns** for fast path

## Dynamic Agent Creation

### Meta-Planning Process

```python
# roles/core_planning.py

class PlanningRole:
    """Meta-planner that designs custom agents"""

    NAME = "planning"
    MODEL = "sonnet"  # Strong model for meta-planning
    REQUIRED_TOOLS = []  # Planning doesn't use tools directly

    async def plan_and_configure_agent(
        self,
        request: str,
        context: TaskContext
    ) -> AgentConfiguration:
        """
        Analyze request and design custom agent

        Returns configuration for runtime agent factory
        """

        # Get ALL available tools
        all_tools = self.tool_registry.format_for_llm()

        prompt = f"""
You are a meta-planner designing a custom AI agent.

USER REQUEST: {request}

CONTEXT:
- Time: {context.current_time}
- Location: {context.location}
- Recent activity: {context.recent_log_summary}

{all_tools}

YOUR TASK:
1. Analyze the request
2. Create step-by-step plan
3. Select ONLY tools actually needed (be selective!)
4. Design agent's system prompt

Respond with JSON:
{{
    "analysis": "What the request needs",
    "plan": "Step-by-step natural language plan",
    "selected_tools": ["category.tool_name", ...],
    "agent_system_prompt": "System prompt defining agent behavior",
    "guidance_notes": "Specific guidance or constraints",
    "max_iterations": 10
}}
"""

        result = await self.llm_invoke(prompt, model="sonnet", response_format="json")

        return AgentConfiguration(
            plan=result.plan,
            system_prompt=result.agent_system_prompt,
            tool_names=result.selected_tools,
            guidance=result.guidance_notes,
            max_iterations=result.max_iterations,
            metadata={"analysis": result.analysis}
        )
```

### Runtime Agent Factory

```python
# llm_provider/runtime_agent_factory.py

class RuntimeAgentFactory:
    """Creates custom Strands agents at runtime"""

    async def create_agent(
        self,
        config: AgentConfiguration,
        context: TaskContext
    ) -> Tuple[Agent, IntentCollector]:
        """Create agent with specific tools and configuration"""

        # Create intent collector
        intent_collector = IntentCollector()

        # Load selected tools
        tools = self.tool_registry.get_tools(config.tool_names)

        # Build system prompt
        system_prompt = f"""
{config.system_prompt}

PLAN TO FOLLOW:
{config.plan}

GUIDANCE:
{config.guidance}

CONTEXT:
- Time: {context.current_time}
- Location: {context.location}

You have {config.max_iterations} iterations to complete the task.
Save important information using memory.save tool.
"""

        # Create Strands agent
        agent = Agent(
            model=self.llm_factory.get_model("sonnet"),
            system_prompt=system_prompt,
            tools=tools,
            max_iterations=config.max_iterations,
            enable_planning=True,
            enable_reflection=True,
        )

        return agent, intent_collector
```

### Simplified Workflow Engine

```python
# supervisor/simplified_workflow_engine.py

class SimplifiedWorkflowEngine:
    """Simplified: Plan → Create → Execute → Process Intents"""

    async def execute_complex_request(
        self,
        request: str,
        context: TaskContext
    ) -> WorkflowResult:
        """Execute using dynamic agent"""

        # Step 1: Meta-planning
        agent_config = await self.planning_role.plan_and_configure_agent(
            request, context
        )

        # Step 2: Create runtime agent
        runtime_agent, intent_collector = await self.agent_factory.create_agent(
            agent_config, context
        )

        # Step 3: Set intent collector
        set_current_collector(intent_collector)

        try:
            # Step 4: Run agent autonomously
            result = await runtime_agent.run(request, context_data=context.to_dict())

            # Step 5: Collect intents
            intents = intent_collector.get_intents()

            # Step 6: Process intents
            await self.intent_processor.process_intents(intents)

            # Step 7: Save execution record
            await self._save_execution_record(request, agent_config, result, intents)

            return WorkflowResult(
                response=result.final_output,
                intents=intents,
                metadata={
                    "tools_selected": agent_config.tool_names,
                    "tools_used": result.tools_called,
                    "iterations": result.iteration_count,
                }
            )
        finally:
            clear_current_collector()
```

**What We Removed:**
- ❌ TaskGraph (DAG execution)
- ❌ Task result sharing
- ❌ Progressive summarization
- ❌ Checkpointing
- ❌ Complex workflow coordination

## Intent System Evolution

### Key Insight: Not All Tools Need Intents

**Query Tools (Direct Execution):**
- Read-only, no side effects
- Return results immediately
- Examples: search_memories, get_weather, list_events

**Action Tools (Intent-Based):**
- Write operations, side effects
- Generate intents for later processing
- Examples: save_memory, send_notification, control_device

### Intent Collection

```python
# common/intent_collector.py

class IntentCollector:
    """Collects intents during agent execution"""

    def __init__(self):
        self._intents: List[Intent] = []

    def register(self, intent: Intent):
        self._intents.append(intent)

    def get_intents(self) -> List[Intent]:
        return self._intents.copy()

# Context-local storage
_current_collector: contextvars.ContextVar[Optional[IntentCollector]] = \
    contextvars.ContextVar('intent_collector', default=None)

async def register_intent(intent: Intent):
    """Tools call this to register intents"""
    collector = _current_collector.get()
    if collector:
        collector.register(intent)
```

### Tool Implementation Examples

**Query Tool (Direct):**
```python
async def search_memories(query: str, limit: int = 5) -> str:
    """Search memories - executes directly"""
    results = await memory_provider.search(query, limit)
    return format_results(results)
```

**Action Tool (Intent):**
```python
async def save_memory(
    content: str,
    memory_type: str,
    importance: float = 0.5
) -> str:
    """Save memory - generates intent"""
    intent = MemoryIntent(
        memory_type=memory_type,
        content=content,
        importance=importance,
        timestamp=datetime.now()
    )
    await register_intent(intent)
    return f"Memory will be saved: {content[:50]}..."
```

### Benefits Preserved
- ✅ Single event loop coordination
- ✅ Observability (all actions logged)
- ✅ Testability (verify intents)
- ✅ Batching and deduplication

## Fast-Path Routing

### Router Logic
```python
# roles/core_router.py

async def route_request(self, request: str, context: TaskContext) -> RoutingResponse:
    """Determine routing: fast path vs meta-planning"""

    analysis = await self.llm_invoke(
        self._build_routing_prompt(request, context),
        model="haiku",
        response_format="json"
    )

    if analysis.confidence >= 0.95:
        # HIGH - Direct to predefined role
        return RoutingResponse(
            route=analysis.route,
            confidence=analysis.confidence,
            approach="direct"
        )
    else:
        # LOW - Use meta-planning
        return RoutingResponse(
            route="planning",
            confidence=analysis.confidence,
            approach="meta_planning"
        )
```

### Execution Flow
```python
# supervisor/supervisor.py

async def process_request(self, request: str, context: TaskContext) -> Response:
    """Main entry point"""

    # Route the request
    routing = await self.router_role.route_request(request, context)

    if routing.approach == "direct":
        # FAST PATH (~600ms)
        role = self.role_registry.get_role(routing.route)
        result = await role.execute(request, context)
        return Response(text=result.response)

    elif routing.approach == "meta_planning":
        # COMPLEX PATH (~5-12s)
        result = await self.workflow_engine.execute_complex_request(request, context)
        return Response(text=result.response)
```

**Performance:**
- Fast path: ~600ms (95% of requests)
- Complex path: ~5-12s (5% of requests)

## Home Assistant Integration

### Architecture
Pure integration (not addon) - runs in HA's Python environment:

```
custom_components/universal_agent/
├── __init__.py              # Setup, runs supervisor
├── manifest.json
├── config_flow.py
├── conversation.py         # Conversation agent
├── supervisor/             # Full system
├── roles/                  # All roles
├── llm_provider/          # Tool registry, agent factory
└── common/                # Intents, utilities
```

### Integration Lifecycle
```python
# custom_components/universal_agent/__init__.py

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up Universal Agent"""

    # Create and start supervisor
    supervisor = Supervisor(build_config_from_entry(entry))
    await supervisor.initialize()
    await supervisor.start()

    # Store in hass.data
    hass.data[DOMAIN] = {"supervisor": supervisor}

    # Start continuous heartbeat
    hass.async_create_task(run_continuous_heartbeat(hass, supervisor))

    # Register conversation agent
    await hass.config_entries.async_forward_entry_setup(entry, "conversation")

    return True

async def run_continuous_heartbeat(hass: HomeAssistant, supervisor: Supervisor):
    """Run multi-horizon heartbeat 24/7"""
    while True:
        await supervisor.heartbeat_tick()
        await asyncio.sleep(15 * 60)  # 15 minutes
```

### Conversation Agent
```python
# custom_components/universal_agent/conversation.py

class UniversalAgentConversation(AbstractConversationAgent):
    """Bridges HA voice pipeline to Universal Agent"""

    async def async_process(self, user_input):
        """Called when user speaks/types"""
        supervisor = self.hass.data[DOMAIN]["supervisor"]
        context = await self._build_context_from_ha(user_input)

        result = await supervisor.process_request(
            request=user_input.text,
            context=context
        )

        return {"response": result.text}
```

**Benefits:**
- Easy install via HACS
- Works on all HA installation types
- Native voice pipeline integration
- 24/7 autonomous operation
- Access to HA devices and state

## Long-Term Executive Planning

### Multi-Horizon Heartbeat
```python
# supervisor/supervisor.py

async def heartbeat_tick(self):
    """Multi-horizon heartbeat - different cycles"""

    now = datetime.now()

    # Always: Proactive check (every 15 min)
    await self.proactive_agent.immediate_check()

    # Daily: If 24h passed
    if now - self.last_daily_review > timedelta(hours=24):
        await self.executive_planner.daily_review()
        self.last_daily_review = now

    # Weekly: If 7 days passed
    if now - self.last_weekly_review > timedelta(days=7):
        await self.executive_planner.weekly_review()
        self.last_weekly_review = now

    # Monthly: If 30 days passed
    if now - self.last_monthly_review > timedelta(days=30):
        await self.executive_planner.monthly_review()
        self.last_monthly_review = now
```

### Planning Data Structures
```python
@dataclass
class Goal:
    """Long-term objective (3-12+ months)"""
    id: str
    title: str
    description: str
    category: str
    status: str
    success_criteria: List[str]
    importance: float  # 0.0-1.0
    current_progress: float  # 0.0-1.0
    related_projects: List[str]

@dataclass
class Project:
    """Medium-term initiative (2-12 weeks)"""
    id: str
    title: str
    goal_id: Optional[str]
    status: str
    milestones: List[Milestone]
    next_actions: List[str]
    progress_notes: List[ProgressNote]
```

### Planning Tools
```python
# roles/planning/tools.py

def create_planning_tools(planning_provider) -> List[Tool]:
    # Query tools
    async def list_goals(status: str = "active") -> str:
        """List goals filtered by status"""
        goals = await planning_provider.get_goals(status=status)
        return format_goals(goals)

    # Action tools
    async def create_goal(title: str, description: str, success_criteria: List[str]) -> str:
        """Create new long-term goal"""
        intent = PlanningIntent(action="create_goal", data={...})
        await register_intent(intent)
        return f"Goal created: {title}"

    async def record_progress(project_id: str, summary: str, progress_percent: float) -> str:
        """Record project progress"""
        intent = PlanningIntent(action="record_progress", data={...})
        await register_intent(intent)
        return f"Progress recorded"

    return [
        Tool.from_function(list_goals),
        Tool.from_function(create_goal),
        Tool.from_function(record_progress),
    ]
```

## File References (Templates)

Use these existing files as templates when implementing new components:

**For Tool Modules:**
- `roles/shared_tools/memory_tools.py` - Example tool module structure
- `roles/shared_tools/redis_tools.py` - Tool creation pattern
- Follow LLM-friendly principles from `docs/64_LLM_FRIENDLY_REFACTORING_PRINCIPLES.md`

**For Roles:**
- `roles/core_weather.py` - Current role structure (to be migrated)
- `roles/core_calendar.py` - Another role example
- `roles/core_timer.py` - Role with lifecycle methods

**For Registry Pattern:**
- `llm_provider/role_registry.py` - Existing registry pattern to follow
- `llm_provider/tool_registry.py` - New but follow similar pattern

**For Intent System:**
- `common/intents.py` - Existing intent types
- `common/intent_processor.py` - How intents are processed
- `common/message_bus.py` - Event-driven pattern

**For Agent Execution:**
- `llm_provider/universal_agent.py` - Current agent invocation pattern
- `supervisor/workflow_engine.py` - Current workflow (to be replaced)

**For Testing:**
- `tests/llm_provider/test_role_registry.py` - Registry test pattern
- `tests/supervisor/test_workflow_engine.py` - Workflow test pattern
- `tests/unit/test_intents.py` - Intent testing pattern

## Implementation Plan

### Phase 1: Foundation (Week 1)

**Step 1.1: Create Domain Directory Structure**
```bash
# Create new domain directories
mkdir -p roles/weather roles/calendar roles/timer roles/smart_home
mkdir -p roles/memory roles/search roles/notification roles/planning
touch roles/weather/{__init__.py,tools.py,role.py}
touch roles/calendar/{__init__.py,tools.py,role.py}
touch roles/timer/{__init__.py,tools.py,role.py}
touch roles/smart_home/{__init__.py,tools.py,role.py}
touch roles/memory/{__init__.py,tools.py}
touch roles/search/{__init__.py,tools.py}
touch roles/notification/{__init__.py,tools.py}
touch roles/planning/{__init__.py,tools.py}
```

**Step 1.2: Implement ToolRegistry**
- Create `llm_provider/tool_registry.py`
- Reference: Follow pattern from `llm_provider/role_registry.py`
- Must implement:
  - `__init__()` - Initialize empty registries
  - `initialize(config, providers)` - Load tools from modules
  - `_load_tools_from_module(module_path, provider)` - Dynamic import
  - `_register_tools(category, tools)` - Index tools by name
  - `get_tool(name)`, `get_tools(names)`, `get_all_tools()`
  - `format_for_llm()` - Format all tools for meta-planning
- Imports needed: `importlib`, `inspect`, `strands.Tool`

**Step 1.3: Implement IntentCollector**
- Create `common/intent_collector.py`
- Must implement:
  - `IntentCollector` class with `register()`, `get_intents()`, `clear()`
  - Context-local storage using `contextvars.ContextVar`
  - `set_current_collector()`, `get_current_collector()`, `clear_current_collector()`
  - `register_intent(intent)` - Helper function tools call
- Imports needed: `contextvars`, `List`, `Optional`

**Step 1.4: Create AgentConfiguration**
- Create `common/agent_configuration.py`
- Define `@dataclass` with fields:
  - `plan: str`
  - `system_prompt: str`
  - `tool_names: List[str]`
  - `guidance: str`
  - `max_iterations: int`
  - `metadata: Dict[str, Any]`

**Step 1.5: Implement RuntimeAgentFactory**
- Create `llm_provider/runtime_agent_factory.py`
- Reference: Look at `llm_provider/universal_agent.py` for agent creation patterns
- Must implement:
  - `__init__(tool_registry, llm_factory)`
  - `create_agent(config, context)` - Returns `(Agent, IntentCollector)`
  - `_load_tools(tool_names)` - Load from registry
  - `_build_system_prompt(config, context)` - Combine plan + guidance + context
- Imports needed: `strands.Agent`, `IntentCollector`, `AgentConfiguration`

**Validation:**
```python
# Test tool registry loads tools
registry = ToolRegistry()
await registry.initialize(config, providers)
assert len(registry.get_all_tools()) > 0

# Test intent collector
collector = IntentCollector()
intent = MemoryIntent(...)
collector.register(intent)
assert len(collector.get_intents()) == 1

# Test agent factory creates agent
factory = RuntimeAgentFactory(registry, llm_factory)
config = AgentConfiguration(...)
agent, collector = await factory.create_agent(config, context)
assert agent is not None
```

### Phase 2: Tool Migration (Week 2)

**Step 2.1: Create Weather Domain**
- Create `roles/weather/tools.py`
- Extract tool functions from `roles/core_weather.py`
- Pattern: `def create_weather_tools(weather_provider) -> List[Tool]:`
- Each tool function is async, has docstring, returns str
- Helper functions (format_weather, etc.) stay in same file but aren't tools
- Use `Tool.from_function(func, category="weather")` to create tools
- Mark query vs action tools clearly in comments

- Create `roles/weather/role.py`
- Copy class structure from `roles/core_weather.py`
- Update to new pattern:
  - Add `REQUIRED_TOOLS = ["weather.get_current", ...]` class variable
  - Remove tool definitions (now in tools.py)
  - Add `tool_registry` to `__init__`
  - Implement `initialize()` to load tools from registry
  - Update `execute()` to use intent collector pattern

- Update `roles/weather/__init__.py`:
```python
from .tools import create_weather_tools
from .role import WeatherRole
__all__ = ["create_weather_tools", "WeatherRole"]
```

**Step 2.2: Create Calendar Domain**
- Follow same pattern as weather
- Extract from `roles/core_calendar.py`
- Tools: list_events (query), create_event (action), update_event (action), delete_event (action)
- Action tools must use `await register_intent(CalendarIntent(...))`

**Step 2.3: Create Timer Domain**
- Extract from `roles/core_timer.py`
- Tools: list_timers (query), set_timer (action), cancel_timer (action)
- Timer tools use Redis, so provider is `providers.redis`

**Step 2.4: Create Smart Home Domain**
- Extract from `roles/core_smart_home.py`
- Tools: list_devices (query), get_device_state (query), control_device (action)
- Use Home Assistant provider

**Step 2.5: Create Utility Domains (No Roles)**
- `roles/memory/tools.py` - Extract from `roles/shared_tools/memory_tools.py`
- `roles/search/tools.py` - Extract from existing search tools
- `roles/notification/tools.py` - Create notification tools
- These don't have role.py, only tools.py

**Step 2.6: Update Tool Registry Initialization**
- Update `llm_provider/tool_registry.py` initialize() method
- Add all domain modules to load list
- Test each module loads correctly

**Validation:**
```bash
# Run tool loading test
python -c "from llm_provider.tool_registry import ToolRegistry; import asyncio; asyncio.run(ToolRegistry().initialize(config, providers))"

# Check specific tools exist
python -c "registry.get_tool('weather.get_forecast')"
python -c "registry.get_tool('calendar.create_event')"

# Verify format_for_llm() output
python -c "print(registry.format_for_llm())"
```

### Phase 3: Role Updates (Week 3)

**Step 3.1: Update Role Registry**
- Modify `llm_provider/role_registry.py`
- Update initialization order: Tool registry must be loaded first
- Pass `tool_registry` to role constructors
- Update role discovery to look in domain directories: `roles/*/role.py`

**Step 3.2: Update Each Predefined Role**
For each role (weather, calendar, timer, smart_home):
- Remove inline tool definitions
- Add `REQUIRED_TOOLS` class variable with tool names
- Update `__init__` to accept `tool_registry`
- Add `initialize()` method to load tools from registry
- Update `execute()` to use intent collector:
  - Create collector before execution
  - Set as current collector
  - Run agent
  - Collect intents after
  - Process intents
  - Clear collector in finally block

**Step 3.3: Update Conversation Role**
- `roles/core_conversation.py` is multi-domain
- Update `REQUIRED_TOOLS` to reference tools from multiple domains:
  ```python
  REQUIRED_TOOLS = [
      "memory.search",
      "memory.save",
      "calendar.list_events",  # Opportunistic
      "weather.get_current",   # Opportunistic
  ]
  ```

**Step 3.4: Update Supervisor Initialization**
- Update `supervisor/supervisor.py`
- Ensure initialization order:
  1. Providers
  2. Tool Registry (needs providers)
  3. Role Registry (needs tool registry)
  4. Agent
  5. Everything else

**Validation:**
```python
# Test role loads tools
role = WeatherRole(tool_registry, llm_factory)
await role.initialize()
assert len(role.tools) == len(role.REQUIRED_TOOLS)

# Test role execution with intent collection
result = await role.execute("What's the weather?", context)
assert result.response is not None

# Test fast path end-to-end
routing = await router.route_request("What's the weather?", context)
assert routing.approach == "direct"
role = role_registry.get_role(routing.route)
result = await role.execute("What's the weather?", context)
assert result.response is not None
```

### Phase 4: Meta-Planning (Week 4)

**Step 4.1: Update Planning Role**
- Modify `roles/core_planning.py`
- Add new method `plan_and_configure_agent(request, context)`
- This is separate from existing planning functionality
- Reference: See meta-planning prompt in this doc
- Must call `tool_registry.format_for_llm()` to get all tools
- LLM response should be JSON with structure defined in AgentConfiguration
- Validate tool names exist in registry before returning config

**Step 4.2: Create Meta-Planning Prompt Template**
- Add to planning role or separate file
- Include:
  - Clear instructions for LLM
  - JSON response format
  - Example response
  - Tool selection principles (fewer is better)
  - Context to include

**Step 4.3: Test Tool Selection Logic**
- Create test cases with various request types
- Verify planning agent selects appropriate tools
- Check that JSON parsing works
- Validate tool names against registry

**Validation:**
```python
# Test meta-planning
planning_role = PlanningRole(tool_registry, llm_factory)
config = await planning_role.plan_and_configure_agent(
    "Plan a mountain trip",
    context
)
assert config.plan is not None
assert len(config.tool_names) > 0
assert all(tool_registry.get_tool(name) for name in config.tool_names)

# Test with various request types
configs = []
for request in test_requests:
    config = await planning_role.plan_and_configure_agent(request, context)
    configs.append(config)
# Verify different requests get different tool selections
```

### Phase 5: Workflow Engine (Week 5)

**Step 5.1: Create SimplifiedWorkflowEngine**
- Create `supervisor/simplified_workflow_engine.py`
- Reference: Look at `supervisor/workflow_engine.py` but much simpler
- Must implement:
  - `__init__(planning_role, agent_factory, intent_processor)`
  - `execute_complex_request(request, context)`
  - `_save_execution_record(...)` - Save to Redis for learning
- Flow: Plan → Create Agent → Run → Collect Intents → Process → Return

**Step 5.2: Add Feature Flag**
- In `config.yaml` add: `ENABLE_NEW_ARCHITECTURE: false`
- In `supervisor/supervisor.py` add conditional:
```python
if config.ENABLE_NEW_ARCHITECTURE:
    self.workflow_engine = SimplifiedWorkflowEngine(...)
else:
    self.workflow_engine = WorkflowEngine(...)  # Old
```

**Step 5.3: Update Supervisor.process_request()**
- Keep routing logic same
- For meta_planning approach, use new workflow engine
- Add logging to compare old vs new (if both running)

**Step 5.4: Test End-to-End**
- Enable new architecture
- Test complex requests through full pipeline
- Verify intents are collected and processed
- Check execution records are saved

**Validation:**
```python
# Test simplified workflow engine
engine = SimplifiedWorkflowEngine(planning_role, agent_factory, intent_processor)
result = await engine.execute_complex_request(
    "Plan a trip and add to calendar",
    context
)
assert result.response is not None
assert len(result.intents) > 0
assert result.metadata["iterations"] > 0

# Test full pipeline
supervisor.config.ENABLE_NEW_ARCHITECTURE = True
response = await supervisor.process_request("Complex request", context)
assert response.text is not None
```

### Phase 6: Testing (Week 6)

**Step 6.1: Unit Tests for New Components**

Create `tests/llm_provider/test_tool_registry.py`:
```python
@pytest.mark.asyncio
async def test_tool_registry_initialization():
    registry = ToolRegistry()
    await registry.initialize(config, providers)
    assert len(registry.get_all_tools()) > 0

@pytest.mark.asyncio
async def test_get_tool_by_name():
    tool = registry.get_tool("weather.get_forecast")
    assert tool is not None

def test_format_for_llm():
    formatted = registry.format_for_llm()
    assert "AVAILABLE TOOLS" in formatted
```

Create `tests/common/test_intent_collector.py`:
```python
def test_intent_collector():
    collector = IntentCollector()
    intent = MemoryIntent(...)
    collector.register(intent)
    assert len(collector.get_intents()) == 1

def test_context_local_storage():
    collector = IntentCollector()
    set_current_collector(collector)
    assert get_current_collector() == collector
```

Create `tests/llm_provider/test_runtime_agent_factory.py`:
```python
@pytest.mark.asyncio
async def test_create_agent():
    factory = RuntimeAgentFactory(tool_registry, llm_factory)
    config = AgentConfiguration(...)
    agent, collector = await factory.create_agent(config, context)
    assert agent is not None
    assert collector is not None
```

**Step 6.2: Integration Tests**

Create `tests/integration/test_dynamic_agents.py`:
```python
@pytest.mark.asyncio
async def test_fast_path_execution():
    # Test weather query goes through fast path
    response = await supervisor.process_request("What's the weather?", context)
    assert response.metadata["approach"] == "direct"
    assert response.metadata["duration_ms"] < 1000

@pytest.mark.asyncio
async def test_complex_path_execution():
    # Test complex request uses meta-planning
    response = await supervisor.process_request(
        "Plan a trip, check weather, add to calendar",
        context
    )
    assert response.metadata["approach"] == "meta_planning"
    assert "plan" in response.metadata
```

**Step 6.3: Performance Tests**

Create `tests/performance/test_latency.py`:
```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_fast_path_latency():
    times = []
    for _ in range(10):
        start = time.time()
        await supervisor.process_request("What's the weather?", context)
        times.append(time.time() - start)
    assert np.percentile(times, 95) < 1.0  # P95 < 1s
```

**Step 6.4: Update Existing Tests**
- Find tests that reference old workflow engine
- Update to work with both old and new
- Or mark as skip if only testing old architecture

**Validation:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/llm_provider/test_tool_registry.py -v
pytest tests/integration/test_dynamic_agents.py -v

# Check coverage
pytest tests/ --cov=llm_provider --cov=supervisor --cov=common
```

### Phase 7: Cleanup (Week 7)

**Step 7.1: Update Documentation**
- Update `README.md` with new architecture overview
- Update architecture diagrams
- Document tool creation pattern
- Document role creation pattern
- Add examples

**Step 7.2: Remove Deprecated Code**
- Set `ENABLE_NEW_ARCHITECTURE = True` as default
- Remove old workflow engine code:
  - `common/task_graph.py`
  - Old methods in `supervisor/workflow_engine.py`
  - Task result sharing logic
  - Progressive summarization
  - Checkpointing code
- Remove feature flag after sufficient testing

**Step 7.3: Clean Up Tests**
- Remove tests for deprecated code
- Keep tests that still apply to new architecture
- Ensure test coverage ≥ 57%

**Step 7.4: Final Validation**
```bash
# Run full test suite
pytest tests/ -v --cov

# Check for unused imports
flake8 --select=F401

# Run linting
make lint

# Verify performance targets met
pytest tests/performance/ -v
```

## Migration Strategy

**Parallel Implementation:**
```python
if config.ENABLE_NEW_ARCHITECTURE:
    result = await self._process_with_dynamic_agents(request, context)
else:
    result = await self._process_with_dag(request, context)
```

**Gradual Cutover:**
1. Weeks 1-2: Implement new, old still active
2. Weeks 3-4: Test new with subset
3. Week 5: Run both, compare
4. Week 6: Switch to new
5. Week 7: Remove old

**Rollback:** Set `ENABLE_NEW_ARCHITECTURE = False`

## Risks and Mitigation

### Risk 1: Planning Quality
**Risk:** Meta-planner selects wrong tools or creates poor plans

**Symptoms:**
- Agent has tools but doesn't use them
- Agent missing required tools for task
- Plan doesn't match request intent
- Tool selection too broad or too narrow

**Mitigation:**
- Start with conservative prompt engineering
- Test with diverse request types during Phase 4
- Save all planning outputs to Redis for analysis
- Monitor tool selection patterns
- Iterate on planning prompt based on failures
- Add validation: check selected tools exist in registry
- Fallback: if confidence in plan is low, use predefined role

**Detection:**
```python
# Log planning quality metrics
logger.info(f"Planning: {len(config.tool_names)} tools selected")
logger.info(f"Analysis: {config.metadata['analysis']}")

# Validation check
if len(config.tool_names) == 0:
    logger.warning("Planning selected zero tools!")
if len(config.tool_names) > 15:
    logger.warning("Planning selected too many tools!")
```

### Risk 2: Performance Regression
**Risk:** System slower than current implementation

**Symptoms:**
- Fast path requests taking > 1s
- Complex path requests timing out
- High P95/P99 latencies
- User complaints about slowness

**Mitigation:**
- Preserve fast-path routing (95%+ of requests)
- Use Haiku for router and predefined roles (fast, cheap)
- Set reasonable max_iterations limits (5 for fast path, 15 for complex)
- Profile hot paths and optimize
- Cache tool registry formatting (don't regenerate every time)
- Monitor latency metrics continuously

**Detection:**
```python
# Add timing to all phases
with timer("routing"):
    routing = await router.route_request(...)

with timer("execution"):
    result = await role.execute(...)

# Alert on P95 thresholds
if p95_latency > 1000:  # ms
    alert("Fast path P95 exceeded target")
```

### Risk 3: Intent System Breakage
**Risk:** Context-local intent collection fails in some scenarios

**Symptoms:**
- Intents not being collected
- IntentCollector is None warnings
- Action tools execute but side effects missing
- Tests pass but production fails

**Mitigation:**
- Thorough testing of intent collector in Phase 1
- Use contextvars correctly (not thread-local)
- Fallback: log warning if no collector available
- Always use try/finally to clear collector
- Test with concurrent requests (ensure context isolation)
- Monitor intent processing logs

**Detection:**
```python
# In register_intent():
collector = get_current_collector()
if collector is None:
    logger.warning(
        f"No intent collector for {intent.type}. "
        f"Intent may not be processed!"
    )
    # Still log intent for debugging
    logger.info(f"Orphaned intent: {intent}")
```

### Risk 4: Tool Naming Conflicts
**Risk:** Tool names conflict or are ambiguous

**Symptoms:**
- Tool registration fails
- Wrong tool gets called
- LLM confused by similar tool names
- Planning selects wrong tools

**Mitigation:**
- Enforce strict naming convention: `category.tool_name`
- Validation during registration (fail if conflict)
- Clear, distinctive tool names
- Use category prefixes consistently
- Document naming conventions
- Code review for new tools

**Detection:**
```python
# In _register_tools():
for tool in tools:
    tool_name = f"{category}.{tool.name}"
    if tool_name in self._tools:
        raise ValueError(
            f"Tool naming conflict: {tool_name} already registered"
        )
    self._tools[tool_name] = tool
```

### Risk 5: Migration Breaks Existing Functionality
**Risk:** Migration takes longer or breaks more than expected

**Symptoms:**
- Tests failing
- Features not working
- Integration issues
- User workflows broken

**Mitigation:**
- Implement in parallel (feature flag)
- Comprehensive test suite before migration
- Test both old and new paths side-by-side
- Gradual cutover (not big bang)
- Clear rollback plan
- Monitor error rates during transition
- Keep old code until new proven stable
- Weekly checkpoints with go/no-go decisions

**Detection:**
```python
# During parallel run, compare results
if config.COMPARE_OLD_NEW:
    old_result = await old_workflow_engine.execute(...)
    new_result = await new_workflow_engine.execute(...)

    if old_result != new_result:
        logger.warning(
            f"Result mismatch: old={old_result}, new={new_result}"
        )
```

### Risk 6: Context-Local Storage Issues
**Risk:** ContextVar doesn't work correctly in all execution contexts

**Symptoms:**
- Intent collector lost between async calls
- Multiple concurrent requests interfere
- Collector not cleared properly

**Mitigation:**
- Test with asyncio extensively
- Test with concurrent requests
- Always use try/finally for cleanup
- Verify each async boundary preserves context
- Reference: Python contextvars documentation
- Consider alternative if issues persist (task-local storage)

**Testing:**
```python
# Test concurrent requests don't interfere
async def test_concurrent_intent_collection():
    async def request1():
        collector = IntentCollector()
        set_current_collector(collector)
        await asyncio.sleep(0.1)
        intent = MemoryIntent(content="request1")
        await register_intent(intent)
        intents = collector.get_intents()
        assert len(intents) == 1
        assert intents[0].content == "request1"

    async def request2():
        collector = IntentCollector()
        set_current_collector(collector)
        await asyncio.sleep(0.1)
        intent = MemoryIntent(content="request2")
        await register_intent(intent)
        intents = collector.get_intents()
        assert len(intents) == 1
        assert intents[0].content == "request2"

    await asyncio.gather(request1(), request2())
```

### Risk 7: Tool Registry Growth
**Risk:** As tools are added, format_for_llm() output becomes too large

**Symptoms:**
- Meta-planning prompts hitting token limits
- Slower planning due to large context
- LLM confused by too many options

**Mitigation:**
- Start with reasonable tool count (~30-50)
- Consider tool categories/grouping
- Could implement smart tool filtering in future
- Monitor prompt sizes
- Use tool descriptions efficiently (concise)

**Future Enhancement:**
```python
# Filter tools by relevance before showing to LLM
def format_for_llm_filtered(self, request: str) -> str:
    """
    Show only relevant tool categories based on request
    E.g., if request mentions "weather", prioritize weather tools
    """
    # Semantic filtering logic
    pass
```

## Examples

### Example 1: Simple Weather (Fast Path)
```
Request: "What's the weather?"
→ Router: 98% confidence → weather role
→ WeatherRole executes (haiku, 5 iterations)
→ Calls get_location_weather()
→ Response: "72°F and sunny"
Time: ~600ms
```

### Example 2: Travel Planning (Meta-Planning)
```
Request: "Plan a weekend mountain trip - check weather, find lodging, add to calendar"
→ Router: 65% confidence → meta-planning
→ Planning agent:
  - Analyzes request
  - Selects tools: weather.get_forecast, search.web_search, calendar.create_event, memory.save
  - Creates plan: "1. Get preferences, 2. Check weather, 3. Search lodging, 4. Present options, 5. Book, 6. Add to calendar"
→ Runtime agent created with 4 tools
→ Agent execution (15 iterations):
  - Asks preferences
  - Checks weather
  - Searches lodging
  - Presents options
  - Gets approval
  - Creates calendar event (intent)
  - Saves trip details (intent)
→ Intent processing: CalendarIntent, MemoryIntent
→ Response: "Booked Mountain Lodge for next weekend"
Time: ~8s
```

## Benefits and Tradeoffs

### Benefits
✅ Flexible - Handle novel requests
✅ Simple - No DAG complexity
✅ Maintainable - Clear domain ownership
✅ Extensible - Add tools easily
✅ Fast - Predefined roles preserved
✅ Safe - Intent system maintained

### Tradeoffs
⚠️ Planning overhead (~2s for complex requests)
⚠️ Less predictable behavior
⚠️ Higher token cost for complex paths
⚠️ Migration effort

**Mitigation:**
- Fast path keeps 95% of requests under 1s
- Save execution records for learning
- Set max_iterations limits
- Gradual migration with rollback

## Success Criteria
- [ ] All existing functionality preserved
- [ ] Fast path < 1000ms (P95)
- [ ] Complex path < 15000ms (P95)
- [ ] Test coverage maintained
- [ ] Can handle novel request combinations
- [ ] Code complexity reduced

## Conclusion

This evolution transforms the system from rigid to flexible while maintaining speed and safety. Key improvements:

1. **Central Tool Registry** - Single source of truth
2. **Dynamic Agents** - Runtime tool selection
3. **Meta-Planning** - LLM designs custom agents
4. **Domain Organization** - Clear ownership
5. **Simplified Workflow** - No DAG complexity

The system handles novel requests while preserving fast-path optimization for common cases.

---

**Status:** DRAFT - Ready for Review
**Last Updated:** 2025-01-20
