# Execution Planning Guide

**Date**: 2025-12-27
**Status**: Production
**Phase**: 4 (Meta-Planning with Structured Execution)

---

## Overview

Execution planning provides **structured guidance** for agents during autonomous execution in Phase 4 meta-planning workflows. Instead of executing blindly, agents receive a step-by-step execution plan that keeps them on track.

This guide explains how execution plans work, how to use them, and how agents can dynamically replan when needed.

---

## What is an Execution Plan?

An **ExecutionPlan** is a type-safe, structured representation of how an agent should execute a complex request. It contains:

- **Plan ID**: Unique identifier for tracking
- **Steps**: Ordered list of tool invocations
- **Dependencies**: Which steps depend on others
- **Status**: Current execution state
- **Reasoning**: Why this plan was created
- **Metadata**: Context and tracking information

### Example Plan

```python
ExecutionPlan(
    plan_id="plan_abc123",
    request="Check weather in Seattle and set timer for 10 minutes",
    selected_tools=["weather.get_current_weather", "timer.set_timer"],
    steps=[
        ExecutionStep(
            step_number=1,
            tool_name="weather.get_current_weather",
            description="Get current weather for Seattle",
            parameters={"location": "Seattle"},
            depends_on=[],  # No dependencies
            status=PlanStatus.PENDING
        ),
        ExecutionStep(
            step_number=2,
            tool_name="timer.set_timer",
            description="Set 10-minute timer",
            parameters={"duration": "10m"},
            depends_on=[1],  # Depends on step 1
            status=PlanStatus.PENDING
        ),
    ],
    reasoning="Sequential execution: first get weather, then set timer",
    created_at=1703700000.0,
    status=PlanStatus.PENDING,
)
```

---

## How Execution Planning Works

### Phase 4 Meta-Planning Flow (with Execution Plans)

```
1. User Request
   ↓
2. Router → Low Confidence → Planning
   ↓
3. Meta-Planning (plan_and_configure_agent)
   ├─ LLM selects tools: ["weather.get_current_weather", "timer.set_timer"]
   ├─ ⭐ Create ExecutionPlan with 2 steps
   ├─ ⭐ Add "planning.replan" to toolset
   └─ Return AgentConfiguration with embedded plan
   ↓
4. SimplifiedWorkflowEngine (execute_complex_request)
   ├─ Create runtime agent with tools
   ├─ ⭐ Format execution plan in agent prompt
   ├─ Agent receives structured step-by-step guidance
   └─ Agent executes autonomously
   ↓
5. Agent Execution (10-15 iterations)
   ├─ Step 1: Call weather.get_current_weather()
   ├─ Step 2: Call timer.set_timer()
   ├─ ⭐ If failure: Call replan() to adjust
   └─ Synthesize final response
   ↓
6. Return Result
```

### Key Enhancement: Structured Guidance

**Before Execution Plans:**

```
Agent prompt: "Check weather in Seattle and set timer for 10 minutes"
Agent: *figures out what to do on its own*
```

**With Execution Plans:**

```
Agent prompt: "Execute the following request using the provided execution plan:

REQUEST: Check weather in Seattle and set timer for 10 minutes

EXECUTION PLAN (ID: plan_abc123):
Reasoning: Sequential execution of 2 selected tools

Steps (2 total):

1. Execute get_current_weather from weather domain
   Tool: weather.get_current_weather
   Status: pending

2. Execute set_timer from timer domain
   Tool: timer.set_timer
   Depends on steps: [1]
   Status: pending

IMPORTANT:
- Follow the execution plan steps in order
- If a step fails, you can call replan() tool to revise the plan

Execute the plan now:"
```

Agent: _has clear structure to follow_

---

## Type-Safe Planning Types

All planning operations use Pydantic models for type safety and validation.

### PlanStatus Enum

```python
class PlanStatus(str, Enum):
    PENDING = "pending"       # Not started
    IN_PROGRESS = "in_progress"  # Currently executing
    COMPLETED = "completed"   # Successfully finished
    FAILED = "failed"        # Execution failed
    REPLANNING = "replanning"  # Being revised
```

### ExecutionStep

**Fields:**

- `step_number: int` - Step sequence (1-indexed)
- `tool_name: str` - Tool to execute (domain.tool format)
- `description: str` - Human-readable step description
- `parameters: dict[str, Any]` - Tool parameters
- `depends_on: list[int]` - Step dependencies
- `status: PlanStatus` - Execution status
- `result: Optional[Any]` - Result after execution
- `error: Optional[str]` - Error message if failed

**Validation Rules:**

- Tool name must follow `domain.tool` format
- Step number must be ≥ 1
- Cannot depend on itself
- Cannot have forward dependencies
- Dependencies must be positive integers

### ExecutionPlan

**Fields:**

- `plan_id: str` - Unique identifier
- `request: str` - Original user request
- `selected_tools: list[str]` - Tools from meta-planning
- `steps: list[ExecutionStep]` - Ordered execution steps
- `reasoning: str` - Why this plan was created
- `created_at: float` - Creation timestamp
- `updated_at: Optional[float]` - Last update timestamp
- `status: PlanStatus` - Overall plan status
- `metadata: dict[str, Any]` - Additional context

**Validation Rules:**

- Must have ≥ 1 step
- Must have ≥ 1 selected tool
- Step numbers must be sequential (1, 2, 3, ...)
- All dependencies must reference existing steps
- No duplicate step numbers
- No gaps in step numbering
- Timestamps cannot be in future

### ReplanRequest

**Fields:**

- `current_plan: ExecutionPlan` - Current execution plan
- `execution_state: dict[str, Any]` - Current execution state
- `reason: str` - Why replanning is needed
- `completed_steps: list[int]` - Steps already completed
- `failed_steps: list[int]` - Steps that failed
- `new_information: Optional[str]` - Additional context

**Validation Rules:**

- Completed/failed steps must exist in plan
- No overlap between completed and failed
- All step numbers must be valid

---

## Planning Tools

### create_execution_plan()

**Purpose**: Create structured execution plan from selected tools

**Signature:**

```python
async def create_execution_plan(
    request: str,
    selected_tools: list[str],
    context: dict[str, Any],
) -> ExecutionPlan
```

**Parameters:**

- `request`: Original user request
- `selected_tools`: Tools selected by meta-planner (e.g., ["weather.get_current_weather"])
- `context`: Additional context (user_id, channel_id, etc.)

**Returns**: Type-safe `ExecutionPlan` object

**Example:**

```python
plan = await create_execution_plan(
    request="Check weather and set timer",
    selected_tools=["weather.get_current_weather", "timer.set_timer"],
    context={"user_id": "user123", "channel_id": "channel456"}
)

# Returns ExecutionPlan with:
# - Unique plan_id
# - 2 steps (weather → timer)
# - Sequential dependencies
# - Metadata with context
```

**Current Implementation**: Simple sequential planning
**Future Enhancement**: LLM-driven plan generation with parameter extraction

---

### replan()

**Purpose**: Revise execution plan based on failures or new information

**Signature:**

```python
async def replan(
    replan_request: ReplanRequest
) -> ExecutionPlan
```

**Parameters:**

- `replan_request`: Request containing current plan, state, and reason

**Returns**: Revised `ExecutionPlan` with updates

**Example:**

```python
# Agent encounters error in step 1
replan_request = ReplanRequest(
    current_plan=original_plan,
    execution_state={"current_step": 1, "error": "API timeout"},
    reason="Weather API failed, need alternative approach",
    completed_steps=[],
    failed_steps=[1],
    new_information="Weather service unavailable"
)

revised_plan = await replan(replan_request)

# Returns plan with:
# - Step 1 marked as FAILED
# - Plan status = REPLANNING
# - Updated timestamp
# - Replan history in metadata
```

**Current Implementation**: Status updates and metadata tracking
**Future Enhancement**: LLM-driven alternative step generation

---

## Agent Usage

### Receiving an Execution Plan

Agents receive execution plans in their prompt:

```
EXECUTION PLAN (ID: plan_abc123):
Reasoning: Sequential execution of 2 selected tools

Steps (2 total):

1. Execute get_current_weather from weather domain
   Tool: weather.get_current_weather
   Status: pending

2. Execute set_timer from timer domain
   Tool: timer.set_timer
   Depends on steps: [1]
   Status: pending
```

### Following the Plan

Agents should:

1. ✅ Execute steps in order
2. ✅ Respect dependencies
3. ✅ Call specified tools
4. ✅ Provide final synthesis

### Replanning on Failure

If a step fails, agent can call `replan()`:

```python
# Agent's internal reasoning:
# "Step 1 failed with weather API error. I should replan."

replan_result = await replan({
    "current_plan": <current_plan>,
    "execution_state": {"current_step": 1},
    "reason": "Weather API timeout error",
    "completed_steps": [],
    "failed_steps": [1],
})

# Agent receives revised plan and continues
```

---

## Implementation Details

### Creating Plans in Meta-Planning

**In `roles/core_planning.py`:**

```python
# After tool selection (Step 8)
agent_config = AgentConfiguration(...)

# Step 9: Create execution plan
planning_tools = tool_registry.get_tools(["planning.create_execution_plan"])
create_plan_tool = planning_tools[0]

execution_plan = await create_plan_tool(
    request=request,
    selected_tools=agent_config.tool_names,
    context={"user_id": context.user_id, ...}
)

# Store in config metadata
agent_config.metadata["execution_plan"] = execution_plan
agent_config.metadata["execution_plan_id"] = execution_plan.plan_id

# Add replan tool
agent_config.tool_names.append("planning.replan")
```

### Using Plans in Execution

**In `supervisor/simplified_workflow_engine.py`:**

```python
# Check for execution plan
if "execution_plan" in agent_config.metadata:
    execution_plan = agent_config.metadata["execution_plan"]

    # Format plan for prompt
    plan_text = self._format_execution_plan(execution_plan)

    # Include in agent prompt
    execution_prompt = f"""Execute the following request:

REQUEST: {request}

EXECUTION PLAN (ID: {execution_plan.plan_id}):
{plan_text}

Follow the plan steps in order..."""

# Run agent with structured prompt
result = await runtime_agent.run(execution_prompt)
```

---

## Testing

### Unit Tests

**Type validation tests** (`tests/unit/test_planning_types.py`):

- 27 comprehensive tests
- 97% code coverage
- Tests all validation rules
- Tests edge cases and errors

**Tool tests** (`tests/unit/test_planning_tools.py`):

- 17 comprehensive tests
- 100% tool coverage
- Tests create_execution_plan()
- Tests replan()
- Tests integration with ToolRegistry

### Integration Tests

**Test execution planning flow:**

```python
# Test complete meta-planning with execution plan
result = await workflow_engine.execute_complex_request(
    request="Check weather and set timer",
    agent_config=agent_config_with_plan,
    context=test_context
)

# Verify execution plan was used
assert "execution_plan" in agent_config.metadata
assert agent_config.metadata["execution_plan_id"].startswith("plan_")

# Verify replan tool was added
assert "planning.replan" in agent_config.tool_names
```

---

## Best Practices

### When Creating Plans

1. **Be specific**: Clear step descriptions
2. **Extract parameters**: Include tool parameters when possible
3. **Sequential by default**: Simple dependencies
4. **Include context**: Add metadata for debugging

### When Executing Plans

1. **Follow structure**: Respect step order
2. **Check dependencies**: Don't skip required steps
3. **Handle failures**: Use replan() when steps fail
4. **Log progress**: Track step completion

### When Replanning

1. **Clear reason**: Explain why replanning
2. **Mark status**: Update completed/failed steps
3. **Preserve state**: Don't lose completed work
4. **Track history**: Add to replan_history metadata

---

## Common Patterns

### Pattern 1: Sequential Execution

Most common pattern - execute steps in order:

```python
Step 1: weather.get_current_weather → depends_on=[]
Step 2: timer.set_timer → depends_on=[1]
Step 3: notification.send → depends_on=[2]
```

### Pattern 2: Parallel with Merge

Independent steps, then merge:

```python
Step 1: weather.get_current_weather → depends_on=[]
Step 2: search.web_search → depends_on=[]
Step 3: summarization.summarize → depends_on=[1, 2]
```

### Pattern 3: Conditional with Fallback

Try approach, replan on failure:

```python
Initial plan:
  Step 1: weather.get_current_weather → depends_on=[]

If Step 1 fails:
  Call replan() → Alternative approach
  Step 1: search.web_search (fallback) → depends_on=[]
```

---

## Error Handling

### Graceful Degradation

**If planning tools unavailable:**

```python
# System logs warning and continues
logger.warning("Planning tools not available, skipping execution plan creation")

# Agent executes without structured plan
# Still works, just without guidance
```

**If plan creation fails:**

```python
except Exception as e:
    logger.error(f"Failed to create execution plan: {e}")
    logger.warning("Continuing without structured execution plan")
    # Agent proceeds with AgentConfiguration only
```

### Replan on Failure

**Agent detects failure:**

```python
# Step execution fails
try:
    result = await weather.get_current_weather("Seattle")
except APIError as e:
    # Agent calls replan tool
    revised_plan = await replan(ReplanRequest(
        current_plan=plan,
        execution_state={"current_step": 1},
        reason=f"Weather API failed: {e}",
        failed_steps=[1],
    ))
    # Continue with revised plan
```

---

## Monitoring and Observability

### Log Markers

**Plan creation:**

```
INFO - roles.planning.tools - Creating execution plan for request: Check weather and...
INFO - roles.planning.tools - Selected tools: ['weather.get_current_weather', 'timer.set_timer']
INFO - roles.planning.tools - Created execution plan plan_abc123 with 2 steps
```

**Plan usage:**

```
INFO - supervisor.simplified_workflow_engine - Using execution plan plan_abc123 with 2 steps
DEBUG - supervisor.simplified_workflow_engine - Plan: [formatted plan text]
```

**Replanning:**

```
INFO - roles.planning.tools - Replanning for plan plan_abc123
INFO - roles.planning.tools - Reason: Weather API failed
INFO - roles.planning.tools - Completed steps: []
INFO - roles.planning.tools - Failed steps: [1]
INFO - roles.planning.tools - Created revised plan for plan_abc123
```

### Metrics to Track

**Plan Metrics:**

- Plans created per workflow
- Average steps per plan
- Plan completion rate
- Replan frequency

**Execution Metrics:**

- Steps completed vs failed
- Average execution time per step
- Dependency satisfaction rate
- Replan success rate

**Quality Metrics:**

- Tool selection accuracy
- Parameter extraction accuracy
- Plan adherence by agents
- Final success rate

---

## API Reference

### Planning Types

**Module**: `common.planning_types`

```python
from common.planning_types import (
    ExecutionPlan,      # Structured execution plan
    ExecutionStep,      # Single step in plan
    PlanStatus,         # Status enum
    ReplanRequest,      # Request to revise plan
)
```

### Planning Tools

**Module**: `roles.planning.tools`

```python
from roles.planning.tools import create_planning_tools

# Create tools
tools = create_planning_tools(planning_provider=None)

# Tools available:
# - create_execution_plan(request, selected_tools, context) -> ExecutionPlan
# - replan(replan_request) -> ExecutionPlan
```

### Accessing in Agent

Agents access planning tools through ToolRegistry:

```python
# Meta-planning adds replan tool
agent_config.tool_names.append("planning.replan")

# Agent can call during execution
revised_plan = await replan({...})
```

---

## Configuration

### Enable Execution Planning

Execution planning is automatically enabled when:

1. ✅ Phase 4 meta-planning is enabled (`enable_phase4_meta_planning: true`)
2. ✅ Planning tools are loaded in ToolRegistry
3. ✅ Request routes to meta-planning pathway (confidence < 0.70)

**In `config.yaml`:**

```yaml
feature_flags:
  enable_phase4_meta_planning: true # Enables entire Phase 4

# Planning tools loaded automatically
# No additional configuration needed
```

### Disable Execution Planning

To run Phase 4 without execution plans:

1. Remove planning tools from ToolRegistry
2. System degrades gracefully (logs warning, continues)
3. Agents execute without structured guidance

---

## Future Enhancements

### Phase 4.1: LLM-Driven Plan Generation

**Current**: Simple sequential plans
**Future**: Intelligent plan generation

```python
# LLM analyzes request and extracts parameters
plan = await create_execution_plan_with_llm(
    request="Check weather in Seattle and set timer for 10 minutes",
    selected_tools=["weather.get_current_weather", "timer.set_timer"],
)

# Returns plan with extracted parameters:
Step 1: weather.get_current_weather
  Parameters: {"location": "Seattle"}  ← Extracted from request

Step 2: timer.set_timer
  Parameters: {"duration": "10m"}  ← Extracted from request
```

### Phase 4.2: Intelligent Replanning

**Current**: Status updates and metadata
**Future**: LLM generates alternative approaches

```python
# LLM analyzes failure and suggests alternatives
revised_plan = await replan_with_llm(
    current_plan=plan,
    failure_context="Weather API timeout",
    available_tools=tool_registry.get_all_tools()
)

# Returns plan with alternative steps:
Original Step 1: weather.get_current_weather (FAILED)
New Step 1: search.web_search(query="weather Seattle")  ← Alternative
```

### Phase 4.3: Parallel Execution

**Current**: Sequential dependencies only
**Future**: Parallel step execution

```python
# Independent steps can run in parallel
Step 1: weather.get_current_weather → depends_on=[]
Step 2: search.web_search → depends_on=[]  # Can run parallel with Step 1
Step 3: summarization.summarize → depends_on=[1, 2]  # Waits for both
```

### Phase 4.4: Conditional Steps

**Current**: All steps execute
**Future**: Conditional execution based on results

```python
Step 1: weather.get_current_weather
Step 2: IF (temperature < 50°F) THEN timer.set_timer("reminder to bring jacket")
Step 3: ELSE notification.send("Weather is nice")
```

---

## Troubleshooting

### Plan Not Created

**Symptom**: Agent executes without plan
**Log**: "Planning tools not available, skipping execution plan creation"
**Cause**: Planning tools not loaded in ToolRegistry
**Solution**: Verify planning provider configured in ToolRegistry initialization

### Invalid Plan Generated

**Symptom**: ValidationError during plan creation
**Log**: "Failed to create execution plan: ValidationError..."
**Cause**: Pydantic validation failed (invalid tool names, dependencies, etc.)
**Solution**: Check tool names follow `domain.tool` format, steps are sequential

### Replan Tool Not Working

**Symptom**: Agent can't call replan()
**Log**: "Tool not found: planning.replan"
**Cause**: Replan tool not added to agent's toolset
**Solution**: Verify Step 9 in meta-planning adds replan tool to tool_names

### Agent Ignores Plan

**Symptom**: Agent doesn't follow execution plan steps
**Log**: Agent executes differently than plan specifies
**Cause**: Plan formatting or prompt structure issue
**Solution**: Verify \_format_execution_plan() produces clear output, check agent prompt

---

## Examples

### Example 1: Simple Sequential Plan

**Request**: "Check weather in Portland and set timer for 5 minutes"

**Generated Plan**:

```
Plan ID: plan_7f3a2b01
Reasoning: Sequential execution of 2 selected tools

Steps:
1. Execute get_current_weather from weather domain
   Tool: weather.get_current_weather

2. Execute set_timer from timer domain
   Tool: timer.set_timer
   Depends on: [1]
```

**Execution**:

- Agent calls weather.get_current_weather()
- Agent calls timer.set_timer()
- Agent synthesizes response
- Total time: ~12s

---

### Example 2: Complex Multi-Domain Plan

**Request**: "Search for restaurants in Seattle, check weather, and add dinner to calendar"

**Generated Plan**:

```
Plan ID: plan_9c5d8e42
Reasoning: Sequential execution of 3 selected tools

Steps:
1. Execute web_search from search domain
   Tool: search.web_search

2. Execute get_current_weather from weather domain
   Tool: weather.get_current_weather
   Depends on: [1]

3. Execute add_event from calendar domain
   Tool: calendar.add_event
   Depends on: [1, 2]
```

**Execution**:

- Agent calls search.web_search("restaurants Seattle")
- Agent calls weather.get_current_weather("Seattle")
- Agent calls calendar.add_event(...)
- Agent synthesizes with all information
- Total time: ~15s

---

### Example 3: Replanning After Failure

**Request**: "Check weather in London and set reminder"

**Initial Plan**:

```
Plan ID: plan_4a8b3f12
Steps:
1. weather.get_current_weather
2. timer.set_timer
```

**Execution**:

```
Step 1: Call weather.get_current_weather("London")
Result: ERROR - API timeout

Agent: "Step 1 failed. I'll replan."

Call replan({
    current_plan: plan_4a8b3f12,
    reason: "Weather API timeout",
    failed_steps: [1],
})

Revised Plan:
Plan ID: plan_4a8b3f12 (same ID, revised)
Status: REPLANNING
Updated: 1703700120.0

Steps:
1. weather.get_current_weather (FAILED) - marked failed
2. timer.set_timer (PENDING) - still pending

Agent: "Weather unavailable, but I can still set the reminder."

Step 2: Call timer.set_timer("reminder")
Result: SUCCESS

Final Response: "I've set a reminder for you. Unfortunately, I couldn't
retrieve the weather for London due to a service issue."
```

---

## Summary

Execution planning provides:

- ✅ **Structured guidance** for agent execution
- ✅ **Type-safe plans** with Pydantic validation
- ✅ **Dynamic replanning** on failures
- ✅ **Clear dependencies** between steps
- ✅ **Observability** with plan IDs and status tracking
- ✅ **Graceful degradation** when unavailable

This keeps agents on track during complex multi-domain workflows while allowing adaptive behavior when things go wrong.

---

**See also:**

- [ROUTING_ARCHITECTURE.md](./ROUTING_ARCHITECTURE.md) - When to use Phase 3 vs Phase 4
- [common/planning_types.py](../common/planning_types.py) - Type definitions
- [roles/planning/tools.py](../roles/planning/tools.py) - Tool implementations
- [PHASE4_FINAL_STATUS.md](../PHASE4_FINAL_STATUS.md) - Phase 4 completion status
