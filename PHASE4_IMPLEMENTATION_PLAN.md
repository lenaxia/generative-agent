# Phase 4: Meta-Planning Implementation Plan

**Date:** 2025-12-22
**Status:** Ready to Begin
**Duration:** 4-6 weeks
**Priority:** High

---

## Executive Summary

Implement meta-planning system for dynamic agent creation with runtime tool selection. This replaces the rigid TaskGraph DAG workflow with flexible, autonomous agents that select tools at runtime.

**Starting Point:** Phase 3 complete

- ✅ ToolRegistry implemented and working
- ✅ 5 domain roles with tools
- ✅ Event-driven architecture
- ✅ Current planning role (TaskGraph generation)

**Goal:** Add meta-planning capability alongside current planning

---

## What Phase 4 Adds

### Current Flow (TaskGraph):

```
Request → Router → Planning → TaskGraph → WorkflowEngine → Execute Roles
```

### Phase 4 Flow (Meta-Planning):

```
Request → Router → Meta-Planning → AgentConfiguration → RuntimeAgentFactory → Custom Agent
```

Both will coexist during migration.

---

## Phase 4 Components (What's NEW)

### 1. ✅ Already Complete (Phase 3)

- ✅ ToolRegistry - Central tool registry
- ✅ Domain roles - weather, timer, calendar, smart_home, search
- ✅ Tool loading infrastructure
- ✅ Intent system
- ✅ Event-driven architecture

### 2. ⏳ Need to Implement (Phase 4)

- ⏳ AgentConfiguration dataclass
- ⏳ RuntimeAgentFactory
- ⏳ IntentCollector (context-local)
- ⏳ plan_and_configure_agent() in planning role
- ⏳ SimplifiedWorkflowEngine
- ⏳ Feature flag system
- ⏳ Testing and validation

---

## Implementation Plan

### Week 1: Foundation Components

#### Day 1-2: Core Data Structures

**Task 1.1: Create AgentConfiguration**

```python
# File: common/agent_configuration.py

from dataclasses import dataclass
from typing import Any

@dataclass
class AgentConfiguration:
    """Configuration for dynamically created agents."""
    plan: str                           # Step-by-step execution plan
    system_prompt: str                  # Custom agent system prompt
    tool_names: list[str]              # Selected tools from registry
    guidance: str                       # Specific guidance/constraints
    max_iterations: int                 # Iteration limit
    metadata: dict[str, Any]           # Additional metadata
```

**Task 1.2: Create IntentCollector**

```python
# File: common/intent_collector.py

import contextvars
from typing import List, Optional
from common.intents import Intent

class IntentCollector:
    """Collects intents during agent execution using context-local storage."""

    def __init__(self):
        self._intents: List[Intent] = []

    def register(self, intent: Intent):
        """Register an intent."""
        self._intents.append(intent)

    def get_intents(self) -> List[Intent]:
        """Get all collected intents."""
        return self._intents.copy()

    def clear(self):
        """Clear all intents."""
        self._intents.clear()

# Context-local storage
_current_collector: contextvars.ContextVar[Optional[IntentCollector]] = \
    contextvars.ContextVar('intent_collector', default=None)

def set_current_collector(collector: IntentCollector):
    """Set the current intent collector."""
    _current_collector.set(collector)

def get_current_collector() -> Optional[IntentCollector]:
    """Get the current intent collector."""
    return _current_collector.get()

def clear_current_collector():
    """Clear the current intent collector."""
    _current_collector.set(None)

async def register_intent(intent: Intent):
    """Register an intent with the current collector."""
    collector = get_current_collector()
    if collector:
        collector.register(intent)
    else:
        logger.warning(f"No intent collector active for {intent.__class__.__name__}")
```

**Validation:**

```bash
# Test data structures
python3 -c "from common.agent_configuration import AgentConfiguration; print('✅ AgentConfiguration')"
python3 -c "from common.intent_collector import IntentCollector; print('✅ IntentCollector')"
```

---

#### Day 3-4: RuntimeAgentFactory

**Task 1.3: Implement RuntimeAgentFactory**

```python
# File: llm_provider/runtime_agent_factory.py

import logging
from typing import Any, Tuple
from strands import Agent

from common.agent_configuration import AgentConfiguration
from common.intent_collector import IntentCollector, set_current_collector
from llm_provider.tool_registry import ToolRegistry
from llm_provider.factory import LLMFactory

logger = logging.getLogger(__name__)


class RuntimeAgentFactory:
    """Creates custom Strands agents at runtime with selected tools."""

    def __init__(self, tool_registry: ToolRegistry, llm_factory: LLMFactory):
        """Initialize the runtime agent factory.

        Args:
            tool_registry: Central tool registry
            llm_factory: LLM factory for model creation
        """
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        logger.info("RuntimeAgentFactory initialized")

    async def create_agent(
        self,
        config: AgentConfiguration,
        context: Any
    ) -> Tuple[Agent, IntentCollector]:
        """Create a custom agent with specific tools and configuration.

        Args:
            config: Agent configuration from meta-planning
            context: Task context for prompt building

        Returns:
            Tuple of (Agent instance, IntentCollector)
        """
        logger.info(f"Creating runtime agent with {len(config.tool_names)} tools")

        # Create intent collector
        intent_collector = IntentCollector()

        # Load selected tools from registry
        tools = self.tool_registry.get_tools(config.tool_names)
        logger.info(f"Loaded {len(tools)} tools: {config.tool_names}")

        # Build comprehensive system prompt
        system_prompt = self._build_system_prompt(config, context)

        # Create Strands agent
        agent = Agent(
            model=self.llm_factory.get_model("sonnet"),
            system_prompt=system_prompt,
            tools=tools,
            max_iterations=config.max_iterations,
        )

        logger.info(f"Created runtime agent with {len(tools)} tools, max_iterations={config.max_iterations}")
        return agent, intent_collector

    def _build_system_prompt(self, config: AgentConfiguration, context: Any) -> str:
        """Build system prompt combining plan, guidance, and context.

        Args:
            config: Agent configuration
            context: Task context

        Returns:
            Complete system prompt string
        """
        prompt_parts = [
            config.system_prompt,
            "",
            "PLAN TO FOLLOW:",
            config.plan,
            "",
            "GUIDANCE:",
            config.guidance,
            "",
            "CONTEXT:",
            f"- Time: {getattr(context, 'current_time', 'unknown')}",
            f"- Location: {getattr(context, 'location', 'unknown')}",
            "",
            f"You have {config.max_iterations} iterations to complete the task.",
            "Save important information using memory.save tool when appropriate.",
        ]

        return "\n".join(prompt_parts)
```

**Validation:**

```bash
# Test agent factory
python3 test_runtime_agent_factory.py
```

---

#### Day 5: Feature Flag System

**Task 1.4: Add Feature Flag**

```python
# File: config.py (or wherever config is stored)

# Phase 4 Feature Flag
ENABLE_PHASE4_META_PLANNING = False  # Default off for safety
```

**Task 1.5: Update Supervisor**

```python
# File: supervisor/supervisor.py

async def process_request(self, request: str, context: TaskContext) -> Response:
    """Main entry point with Phase 4 feature flag."""

    # Route the request
    routing = await self.router_role.route_request(request, context)

    if routing.approach == "direct":
        # FAST PATH (~600ms) - unchanged
        role = self.role_registry.get_role(routing.route)
        result = await role.execute(request, context)
        return Response(text=result.response)

    elif routing.approach == "meta_planning":
        # COMPLEX PATH - check feature flag
        if self.config.ENABLE_PHASE4_META_PLANNING:
            # NEW: Use Phase 4 meta-planning
            result = await self.simplified_workflow_engine.execute_complex_request(
                request, context
            )
        else:
            # OLD: Use current TaskGraph planning
            result = await self.workflow_engine.execute_complex_request(
                request, context
            )

        return Response(text=result.response)
```

---

### Week 2: Meta-Planning Logic

#### Day 6-8: Add plan_and_configure_agent()

**Task 2.1: Extend Planning Role**

```python
# File: roles/core_planning.py

async def plan_and_configure_agent(
    self,
    request: str,
    context: TaskContext
) -> AgentConfiguration:
    """NEW: Meta-planning for dynamic agents (Phase 4).

    This is separate from the existing TaskGraph generation.

    Args:
        request: User request to plan for
        context: Task context

    Returns:
        AgentConfiguration for runtime agent creation
    """
    logger.info(f"Phase 4 meta-planning for: {request}")

    # Get ALL available tools from ToolRegistry
    all_tools = self.tool_registry.format_for_llm()

    # Build meta-planning prompt
    prompt = f"""You are a meta-planner designing a custom AI agent.

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

    # Call LLM for meta-planning
    result = await self.llm_invoke(prompt, model="sonnet", response_format="json")

    # Parse and validate
    import json
    planning_result = json.loads(result)

    # Validate tool names exist in registry
    for tool_name in planning_result["selected_tools"]:
        if not self.tool_registry.get_tool(tool_name):
            logger.warning(f"Meta-planning selected unavailable tool: {tool_name}")

    # Create configuration
    return AgentConfiguration(
        plan=planning_result["plan"],
        system_prompt=planning_result["agent_system_prompt"],
        tool_names=planning_result["selected_tools"],
        guidance=planning_result["guidance_notes"],
        max_iterations=planning_result.get("max_iterations", 10),
        metadata={"analysis": planning_result["analysis"]}
    )
```

**Validation:**

```bash
# Test meta-planning
python3 test_meta_planning.py
```

---

#### Day 9-10: SimplifiedWorkflowEngine

**Task 2.2: Create SimplifiedWorkflowEngine**

```python
# File: supervisor/simplified_workflow_engine.py

import logging
from typing import Any
from dataclasses import dataclass

from common.intent_collector import set_current_collector, clear_current_collector
from llm_provider.runtime_agent_factory import RuntimeAgentFactory

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    response: str
    intents: list[Any]
    metadata: dict[str, Any]


class SimplifiedWorkflowEngine:
    """Simplified workflow: Plan → Create → Execute → Process Intents."""

    def __init__(
        self,
        planning_role: Any,
        agent_factory: RuntimeAgentFactory,
        intent_processor: Any
    ):
        """Initialize simplified workflow engine.

        Args:
            planning_role: Planning role with plan_and_configure_agent()
            agent_factory: RuntimeAgentFactory for creating agents
            intent_processor: IntentProcessor for processing intents
        """
        self.planning_role = planning_role
        self.agent_factory = agent_factory
        self.intent_processor = intent_processor
        logger.info("SimplifiedWorkflowEngine initialized")

    async def execute_complex_request(
        self,
        request: str,
        context: Any
    ) -> WorkflowResult:
        """Execute complex request using dynamic agent.

        Flow: Plan → Create Agent → Run → Collect Intents → Process → Return

        Args:
            request: User request
            context: Task context

        Returns:
            WorkflowResult with response and metadata
        """
        logger.info(f"SimplifiedWorkflowEngine executing: {request}")

        # Step 1: Meta-planning
        agent_config = await self.planning_role.plan_and_configure_agent(
            request, context
        )
        logger.info(f"Meta-planning selected {len(agent_config.tool_names)} tools")

        # Step 2: Create runtime agent
        runtime_agent, intent_collector = await self.agent_factory.create_agent(
            agent_config, context
        )

        # Step 3: Set intent collector as current
        set_current_collector(intent_collector)

        try:
            # Step 4: Run agent autonomously
            result = await runtime_agent.run(request)

            # Step 5: Collect intents
            intents = intent_collector.get_intents()
            logger.info(f"Collected {len(intents)} intents from agent execution")

            # Step 6: Process intents
            await self.intent_processor.process_intents(intents)

            # Step 7: Return result
            return WorkflowResult(
                response=result.final_output,
                intents=intents,
                metadata={
                    "approach": "phase4_meta_planning",
                    "tools_selected": agent_config.tool_names,
                    "tools_used": getattr(result, 'tools_called', []),
                    "iterations": getattr(result, 'iteration_count', 0),
                    "plan": agent_config.plan,
                }
            )

        finally:
            # Step 8: Clear intent collector
            clear_current_collector()
```

**Validation:**

```bash
# Test simplified workflow engine
python3 test_simplified_workflow_engine.py
```

---

### Week 3: Integration & Testing

#### Day 11-13: Integration

**Task 3.1: Wire Everything Together**

```python
# File: supervisor/supervisor.py

async def initialize(self):
    """Initialize supervisor with Phase 4 components."""

    # ... existing initialization ...

    # Phase 4: Create RuntimeAgentFactory
    self.agent_factory = RuntimeAgentFactory(
        self.tool_registry,
        self.llm_factory
    )

    # Phase 4: Create SimplifiedWorkflowEngine
    self.simplified_workflow_engine = SimplifiedWorkflowEngine(
        self.role_registry.get_role("planning"),
        self.agent_factory,
        self.intent_processor
    )

    logger.info("Phase 4 components initialized")
```

**Task 3.2: End-to-End Test**

```python
# File: test_phase4_integration.py

async def test_phase4_end_to_end():
    """Test complete Phase 4 flow."""

    # Setup
    supervisor = Supervisor(config)
    supervisor.config.ENABLE_PHASE4_META_PLANNING = True
    await supervisor.initialize()

    # Test complex request
    request = "Plan a mountain trip - check weather, search lodging, add to calendar"
    response = await supervisor.process_request(request, context)

    # Verify
    assert response.text is not None
    assert "mountain" in response.text.lower()
    assert response.metadata["approach"] == "phase4_meta_planning"
    assert len(response.metadata["tools_selected"]) > 0
```

---

#### Day 14-15: Testing

**Task 3.3: Comprehensive Tests**

1. **Unit Tests:**

   - `test_agent_configuration.py`
   - `test_intent_collector.py`
   - `test_runtime_agent_factory.py`
   - `test_meta_planning.py`
   - `test_simplified_workflow_engine.py`

2. **Integration Tests:**

   - `test_phase4_integration.py`
   - `test_phase4_with_real_tools.py`
   - `test_phase4_vs_taskgraph.py` (compare outputs)

3. **Performance Tests:**
   - `test_phase4_latency.py`
   - `test_phase4_tool_selection_quality.py`

---

### Week 4: Validation & Refinement

#### Day 16-18: Validation

**Task 4.1: Side-by-Side Comparison**

```python
# Run both systems on same requests, compare:
# - Response quality
# - Tool selection
# - Execution time
# - Intent processing
```

**Task 4.2: Edge Cases**

- Empty tool selection
- Invalid tool names
- Timeout scenarios
- Intent processing failures

---

#### Day 19-20: Documentation

**Task 4.3: Update Documentation**

- `docs/PHASE4_META_PLANNING.md` - Architecture overview
- `docs/PHASE4_MIGRATION_GUIDE.md` - Migration steps
- `CLAUDE.md` - Update with Phase 4 patterns
- `README.md` - Update architecture section

**Task 4.4: Create Examples**

- Example meta-planning requests
- Example agent configurations
- Example custom agents

---

### Week 5-6: Gradual Rollout

#### Week 5: Parallel Testing

**Task 5.1: Enable for Test Instances**

```python
# In config.py
ENABLE_PHASE4_META_PLANNING = True  # Enable in test environment
```

**Task 5.2: Monitor and Log**

- Log all Phase 4 executions
- Compare with TaskGraph results
- Collect metrics

**Task 5.3: Iterate on Prompts**

- Refine meta-planning prompt
- Improve tool selection logic
- Optimize agent system prompts

---

#### Week 6: Production Rollout

**Task 6.1: Enable by Default**

```python
# In config.py
ENABLE_PHASE4_META_PLANNING = True  # Default for all instances
```

**Task 6.2: Monitor Production**

- Error rates
- Response quality
- User satisfaction
- Performance metrics

**Task 6.3: Cleanup (after validation)**

- Remove old WorkflowEngine code
- Remove TaskGraph generation
- Remove feature flag
- Update tests

---

## Success Criteria

### Functional Requirements

- ✅ Meta-planning selects appropriate tools
- ✅ RuntimeAgentFactory creates working agents
- ✅ Agents execute autonomously
- ✅ Intents collected and processed
- ✅ Response quality matches or exceeds TaskGraph

### Performance Requirements

- ✅ Fast path unchanged (~600ms, 95% of requests)
- ✅ Complex path < 15s (P95)
- ✅ Meta-planning overhead < 3s
- ✅ Tool selection accurate (>90%)

### Quality Requirements

- ✅ All tests pass (100%)
- ✅ No regressions
- ✅ Clean rollback path
- ✅ Comprehensive logging
- ✅ Clear error messages

---

## Risk Mitigation

### Risk 1: Poor Tool Selection

**Mitigation:**

- Extensive prompt engineering
- Validation of tool names
- Fallback to TaskGraph on errors
- Monitoring and alerting

### Risk 2: Performance Regression

**Mitigation:**

- Performance tests before rollout
- Gradual rollout with monitoring
- Fast path preserved (no changes)
- Feature flag for instant rollback

### Risk 3: Agent Execution Failures

**Mitigation:**

- Comprehensive error handling
- Timeout limits (max_iterations)
- Intent collection in try/finally
- Detailed logging for debugging

---

## Next Steps

**Immediate (Day 1):**

1. Create `common/agent_configuration.py`
2. Create `common/intent_collector.py`
3. Test data structures

**This Week:**

1. Implement RuntimeAgentFactory
2. Add feature flag
3. Create basic tests

**This Month:**

1. Implement meta-planning
2. Create SimplifiedWorkflowEngine
3. Integration testing
4. Documentation

---

**Status:** Ready to begin Phase 4 implementation
**Estimated Duration:** 4-6 weeks
**Priority:** High
**Confidence:** High (Phase 3 foundation solid)
