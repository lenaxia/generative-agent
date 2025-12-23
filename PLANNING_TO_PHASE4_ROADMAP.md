# Planning Role Evolution: Current → Phase 4 Meta-Planning

**Date:** 2025-12-22
**Status:** Analysis Complete, Roadmap Defined

---

## Executive Summary

You have **TWO DIFFERENT types of planning**:

1. **Current Planning (`core_planning.py`)**: Workflow orchestration using TaskGraph DAGs
2. **Phase 4 Meta-Planning (Doc 65)**: Dynamic agent creation with runtime tool selection

These are **NOT the same thing**. The current planning role should be **kept** while Phase 4 is implemented in parallel.

---

## Current Planning Role (core_planning.py)

### What It Does

**Purpose:** Generate multi-step TaskGraph workflows using predefined roles

**Architecture:** TaskGraph DAG → WorkflowEngine Execution

### Key Features

```python
ROLE_CONFIG = {
    "name": "planning",
    "llm_type": "STRONG",
    "fast_reply": False,  # Multi-step planning
    "exclude_from_planning": True,  # Don't use planning within planning
}
```

**Pre-Processing:**
- `load_planning_context()` - Loads dual-layer memory (realtime + assessed)
- `load_available_roles()` - Discovers available roles from RoleRegistry

**LLM Task:**
- Analyzes user request
- Creates TaskGraph JSON following BNF grammar:
  ```json
  {
    "tasks": [
      {
        "id": "task_1",
        "name": "Get Weather",
        "description": "Check current weather",
        "role": "weather",
        "parameters": {"location": "Seattle"}
      }
    ],
    "dependencies": [
      {
        "source_task_id": "task_1",
        "target_task_id": "task_2",
        "type": "sequential"
      }
    ]
  }
  ```

**Post-Processing:**
- `validate_task_graph()` - Validates JSON structure and role references
- `execute_task_graph()` - Creates WorkflowIntent
- `save_planning_result()` - Saves to realtime log

**Execution Flow:**
```
User Request
    ↓
Router (confidence < 70%)
    ↓
Planning Role
    ↓
Generate TaskGraph JSON
    ↓
Validate Structure
    ↓
Create WorkflowIntent
    ↓
WorkflowEngine Executes DAG
    ↓
Execute Role A → Role B → Role C (sequential/parallel)
    ↓
Task result sharing
    ↓
Progressive summarization
    ↓
Return Response
```

### Current Role Purpose

✅ **Multi-step workflows** using predefined roles
✅ **Sequential/parallel task execution** with dependencies
✅ **Task result sharing** between steps
✅ **Predefined role selection** (weather, timer, calendar, etc.)

❌ **NOT flexible** - can only use predefined roles
❌ **NOT dynamic** - fixed role capabilities
❌ **Complex** - DAG execution, checkpointing, result sharing

---

## Phase 4 Meta-Planning (Document 65)

### What It Is

**Purpose:** Design and create custom AI agents at runtime with flexible tool selection

**Architecture:** AgentConfiguration → RuntimeAgentFactory → Custom Strands Agent

### Key Features

**Meta-Planning Process:**
```python
async def plan_and_configure_agent(
    self,
    request: str,
    context: TaskContext
) -> AgentConfiguration:
    """
    Analyze request and design custom agent

    Returns configuration for runtime agent factory
    """

    # Get ALL available tools from ToolRegistry
    all_tools = self.tool_registry.format_for_llm()

    # LLM designs custom agent
    result = await self.llm_invoke(prompt, model="sonnet")

    return AgentConfiguration(
        plan="Step-by-step natural language plan",
        system_prompt="System prompt defining agent behavior",
        tool_names=["weather.get_forecast", "memory.save"],  # Selected tools
        guidance="Specific guidance or constraints",
        max_iterations=10
    )
```

**Runtime Agent Factory:**
```python
async def create_agent(
    self,
    config: AgentConfiguration,
    context: TaskContext
) -> Tuple[Agent, IntentCollector]:
    """Create custom Strands agent with selected tools"""

    # Load only selected tools
    tools = self.tool_registry.get_tools(config.tool_names)

    # Create custom agent
    agent = Agent(
        model=self.llm_factory.get_model("sonnet"),
        system_prompt=config.system_prompt,
        tools=tools,
        max_iterations=config.max_iterations,
    )

    return agent, intent_collector
```

**Simplified Workflow Engine:**
```python
async def execute_complex_request(
    self,
    request: str,
    context: TaskContext
) -> WorkflowResult:
    """Simplified: Plan → Create → Execute → Process Intents"""

    # Step 1: Meta-planning
    agent_config = await self.planning_role.plan_and_configure_agent(request, context)

    # Step 2: Create runtime agent
    runtime_agent, intent_collector = await self.agent_factory.create_agent(agent_config, context)

    # Step 3: Run agent autonomously
    result = await runtime_agent.run(request)

    # Step 4: Process intents
    intents = intent_collector.get_intents()
    await self.intent_processor.process_intents(intents)

    return WorkflowResult(response=result.final_output)
```

**Execution Flow:**
```
User Request
    ↓
Router (confidence < 70%)
    ↓
Meta-Planning Agent
    ↓
Analyze request + ALL tools + context
    ↓
Output: plan + tool selection + agent config
    ↓
RuntimeAgentFactory creates custom Strands agent
    ↓
Agent runs autonomously with selected tools (10-15 iterations)
    ↓
Collect intents → Process → Return
```

### Phase 4 Purpose

✅ **Flexible** - any tool combination
✅ **Dynamic** - runtime tool selection
✅ **Simple** - no DAG complexity
✅ **Autonomous** - agent runs independently
✅ **Easy to extend** - just add tools to registry

---

## Key Differences

| Feature | Current Planning | Phase 4 Meta-Planning |
|---------|-----------------|----------------------|
| **Output** | TaskGraph JSON (DAG) | AgentConfiguration |
| **Execution** | WorkflowEngine executes DAG | Single autonomous agent |
| **Roles** | Predefined roles only | No roles - just tools |
| **Tool Selection** | Role bundles (fixed) | Individual tools (flexible) |
| **Flexibility** | Limited to predefined combinations | Any tool combination |
| **Complexity** | High (DAG, checkpointing, result sharing) | Low (single agent run) |
| **Use Case** | Multi-step workflows with role coordination | Complex single-agent tasks |
| **Architecture** | Multiple role invocations | Single agent invocation |

---

## The Gap: How to Bridge Them

### Current State (What You Have Now)

```
Router
  ├── High Confidence (≥95%) → Fast Path (Predefined Roles)
  │   ├── Weather Role
  │   ├── Timer Role
  │   ├── Calendar Role
  │   └── Smart Home Role
  │
  └── Low Confidence (<70%) → Complex Path (Current Planning)
      └── Planning Role → TaskGraph → WorkflowEngine
```

### Phase 4 Goal (Where You Want To Be)

```
Router
  ├── High Confidence (≥95%) → Fast Path (Predefined Roles)
  │   ├── Weather Role
  │   ├── Timer Role
  │   ├── Calendar Role
  │   └── Smart Home Role
  │
  └── Low Confidence (<70%) → Complex Path (Meta-Planning)
      └── Meta-Planning Role → AgentConfiguration → RuntimeAgentFactory → Custom Agent
```

### Migration Strategy

**Phase 4.1: Foundation (Parallel Implementation)**

1. **Keep Current Planning Role**
   - Don't delete `core_planning.py`
   - It's the current "complex path" and works
   - Will be replaced gradually, not immediately

2. **Create New Components**
   - `common/agent_configuration.py` - AgentConfiguration dataclass
   - `llm_provider/runtime_agent_factory.py` - Creates custom agents
   - `supervisor/simplified_workflow_engine.py` - New workflow engine

3. **Add Feature Flag**
   ```python
   # In config.yaml
   ENABLE_PHASE4_META_PLANNING: false  # Default off

   # In supervisor
   if config.ENABLE_PHASE4_META_PLANNING:
       workflow_engine = SimplifiedWorkflowEngine(...)
   else:
       workflow_engine = WorkflowEngine(...)  # Current DAG-based
   ```

**Phase 4.2: Implement Meta-Planning**

1. **Add New Method to Planning Role**
   ```python
   # In roles/core_planning.py

   async def plan_and_configure_agent(
       self,
       request: str,
       context: TaskContext
   ) -> AgentConfiguration:
       """NEW: Meta-planning for dynamic agents"""
       # Implementation from Doc 65
       pass
   ```

2. **Keep Existing Method**
   ```python
   # Keep existing TaskGraph generation
   # These methods stay as-is for backward compatibility
   def load_available_roles(...)
   def validate_task_graph(...)
   def execute_task_graph(...)
   ```

**Phase 4.3: Gradual Cutover**

1. **Week 1-2:** Implement new components, old system active
2. **Week 3-4:** Test Phase 4 with feature flag on test instances
3. **Week 5:** Run both systems in parallel, log and compare
4. **Week 6:** Enable Phase 4 by default (can still rollback)
5. **Week 7:** Remove old DAG-based workflow engine

**Phase 4.4: Cleanup**

Once Phase 4 is proven stable:
- Remove old TaskGraph generation methods from planning role
- Remove old WorkflowEngine DAG execution code
- Remove `common/task_graph.py`
- Remove checkpointing and result sharing logic

---

## What Stays vs What Changes

### ✅ Stays (Don't Touch)

- **Fast-Reply Roles** - weather, timer, calendar, smart_home work as-is
- **Router** - routing logic stays the same (confidence-based)
- **Intent System** - unchanged, works same way
- **ToolRegistry** - already implemented (Phase 3 complete)
- **RoleRegistry** - minor updates only
- **Providers** - Redis, Calendars, Weather, HA - unchanged
- **Memory System** - dual-layer memory unchanged

### 🔄 Changes (Phase 4 Work)

- **Planning Role** - Add new `plan_and_configure_agent()` method
- **Workflow Engine** - Replace with SimplifiedWorkflowEngine
- **Execution Model** - From DAG to single autonomous agent
- **Tool Selection** - From role bundles to individual tools
- **Supervisor** - Update process_request() to use new workflow engine

### ❌ Removes Eventually (After Phase 4 Proven)

- TaskGraph JSON generation
- DAG execution logic
- Task result sharing
- Progressive summarization
- Checkpointing
- Old workflow engine complexity

---

## Implementation Roadmap

### Immediate (Now)

1. ✅ Restore `core_planning.py` - **DONE**
2. 🔄 Migrate search role to domain pattern - **IN PROGRESS**
3. ⏳ Test current system still works with planning role

### Phase 4 Implementation (4-6 weeks)

**Week 1: Foundation**
- [ ] Create `common/agent_configuration.py`
- [ ] Create `llm_provider/runtime_agent_factory.py`
- [ ] Create `supervisor/simplified_workflow_engine.py`
- [ ] Add feature flag `ENABLE_PHASE4_META_PLANNING`

**Week 2: Meta-Planning**
- [ ] Add `plan_and_configure_agent()` method to planning role
- [ ] Test tool selection from ToolRegistry
- [ ] Validate AgentConfiguration creation

**Week 3: Integration**
- [ ] Update Supervisor to use new workflow engine with flag
- [ ] Test end-to-end with Phase 4 enabled
- [ ] Compare old vs new results

**Week 4: Testing**
- [ ] Unit tests for new components
- [ ] Integration tests for meta-planning
- [ ] Performance tests (latency targets)

**Week 5: Parallel Run**
- [ ] Run both systems side-by-side
- [ ] Log and compare results
- [ ] Identify any issues

**Week 6: Cutover**
- [ ] Enable Phase 4 by default
- [ ] Monitor error rates
- [ ] Keep rollback option available

**Week 7: Cleanup**
- [ ] Remove old workflow engine code
- [ ] Remove TaskGraph logic
- [ ] Update documentation

---

## Benefits of Phase 4

### For Users

✅ **More flexible** - can handle novel request combinations
✅ **Simpler responses** - agent reasons through problem autonomously
✅ **Better context** - agent has full request context, not fragmented tasks

### For Developers

✅ **Less complexity** - no DAG, checkpointing, result sharing
✅ **Easier to extend** - just add tools, no new roles needed
✅ **Better debugging** - single agent execution trace vs DAG
✅ **Cleaner code** - follows LLM-friendly principles

### For System

✅ **Maintainable** - domain-based tool organization
✅ **Extensible** - ToolRegistry makes adding tools easy
✅ **Observable** - intent system preserved
✅ **Safe** - intent-based action processing unchanged

---

## Risks and Mitigation

### Risk: Planning Quality

**Concern:** Meta-planner might select wrong tools or create poor plans

**Mitigation:**
- Test extensively with diverse requests
- Save all planning outputs for analysis
- Monitor tool selection patterns
- Iterate on planning prompt
- Fallback to predefined role if confidence low

### Risk: Performance

**Concern:** Meta-planning adds overhead

**Mitigation:**
- Fast path unchanged (95% of requests)
- Set max_iterations limits
- Use Haiku for fast path, Sonnet for complex
- Profile and optimize hot paths

### Risk: Migration Complexity

**Concern:** Migration breaks existing functionality

**Mitigation:**
- Parallel implementation with feature flag
- Gradual cutover with monitoring
- Clear rollback plan
- Keep old code until new proven

---

## Decision: What To Do Now

### Recommended Approach

1. **✅ Keep `core_planning.py`** - It's the current complex path, don't delete
2. **🔄 Migrate search role** - Complete Phase 3 consistency
3. **⏳ Document Phase 4 plan** - This document
4. **⏳ Prioritize Phase 4 implementation** - 4-6 week project when ready

### Why Keep Current Planning

The current planning role is:
- ✅ Working production code
- ✅ Handles complex multi-step workflows
- ✅ Used when router confidence < 70%
- ✅ Part of current architecture

Deleting it now would:
- ❌ Break the complex path entirely
- ❌ Leave no fallback for low-confidence requests
- ❌ Require immediate Phase 4 implementation

### Next Steps

1. **Complete search migration** (today)
2. **Test system end-to-end** (today)
3. **Review this roadmap** (you decide timing for Phase 4)
4. **Implement Phase 4 when ready** (4-6 week project)

---

## Questions for You

1. **Timeline:** When do you want to start Phase 4 implementation? (It's a 4-6 week project)

2. **Priority:** Is Phase 4 higher priority than other work, or should it wait?

3. **Testing:** Do you want to test current planning with a complex request to verify it works?

4. **Documentation:** Should I add Phase 4 architecture overview to CLAUDE.md for future reference?

---

**Status:** Ready for decision on Phase 4 timing
**Current Focus:** Complete search migration, test system
**Future Work:** Phase 4 meta-planning (when prioritized)
