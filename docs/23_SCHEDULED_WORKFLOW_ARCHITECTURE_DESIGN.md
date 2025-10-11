# Scheduled Workflow Architecture Design

**Document ID:** 23
**Created:** 2025-10-11
**Status:** Future Investigation
**Priority:** Medium

## Executive Summary

This document explores an alternative architecture where timers, alarms, and scheduled actions are unified under a single **Scheduled Workflow** system. Instead of maintaining separate timer infrastructure, all time-based actions would become workflows with execution delays, simplifying the architecture while maintaining full functionality.

## Current Architecture vs Proposed

### Current: Dual System

```
┌─────────────────┐    ┌─────────────────┐
│ Workflow System │    │ Timer System    │
│                 │    │                 │
│ • Immediate     │    │ • Delayed       │
│ • Complex       │    │ • Simple        │
│ • Multi-step    │    │ • Notifications │
│ • Planning      │    │ • Alarms        │
└─────────────────┘    └─────────────────┘
```

### Proposed: Unified System

```
┌─────────────────────────────────────────┐
│         Scheduled Workflow System       │
│                                         │
│ • Immediate execution (delay=0)         │
│ • Delayed execution (delay=600s)        │
│ • Complex multi-step workflows          │
│ • Simple notifications                  │
│ • Planning integration                  │
│ • Timer/alarm functionality             │
└─────────────────────────────────────────┘
```

## Proposed Architecture

### Core Concept

Replace the timer system with **ScheduledWorkflow** objects that are workflows with execution delays:

```python
# Instead of creating a timer:
timer = Timer(duration=600, action="turn on lights")

# Create a scheduled workflow:
workflow = ScheduledWorkflow(
    instruction="turn on the lights",
    delay=600,  # 10 minutes
    context=current_context,
    execution_time=datetime.now() + timedelta(seconds=600)
)
```

### Implementation Components

#### 1. ScheduledWorkflow Class

```python
class ScheduledWorkflow:
    """Workflow with delayed execution capability."""

    def __init__(
        self,
        instruction: str,
        delay: int = 0,
        execution_time: Optional[datetime] = None,
        context: Optional[dict] = None,
        priority: int = 0
    ):
        self.workflow_id = str(uuid.uuid4())
        self.instruction = instruction
        self.delay = delay
        self.execution_time = execution_time or (datetime.now() + timedelta(seconds=delay))
        self.context = context or {}
        self.priority = priority
        self.status = "scheduled"
```

#### 2. Scheduler Engine

```python
class SchedulerEngine:
    """Manages scheduled workflow execution."""

    def __init__(self, workflow_engine, redis_client):
        self.workflow_engine = workflow_engine
        self.redis_client = redis_client
        self.scheduled_workflows = {}

    async def schedule_workflow(self, scheduled_workflow: ScheduledWorkflow):
        """Schedule a workflow for future execution."""
        # Store in Redis with expiry time
        await self.redis_client.zadd(
            "scheduled_workflows",
            {scheduled_workflow.workflow_id: scheduled_workflow.execution_time.timestamp()}
        )

        # Store workflow data
        await self.redis_client.set(
            f"workflow:{scheduled_workflow.workflow_id}",
            json.dumps(scheduled_workflow.to_dict()),
            ex=scheduled_workflow.delay + 3600  # TTL with buffer
        )

    async def check_and_execute_ready_workflows(self):
        """Check for workflows ready to execute."""
        now = datetime.now().timestamp()
        ready_workflow_ids = await self.redis_client.zrangebyscore(
            "scheduled_workflows", 0, now
        )

        for workflow_id in ready_workflow_ids:
            await self.execute_scheduled_workflow(workflow_id)
```

#### 3. Integration with Existing System

```python
# WorkflowEngine enhancement
class WorkflowEngine:
    def __init__(self, ...):
        # ... existing init ...
        self.scheduler_engine = SchedulerEngine(self, redis_client)

    def start_workflow(self, instruction: str, delay: int = 0, context: dict = None):
        """Start workflow immediately or schedule for later."""
        if delay == 0:
            # Immediate execution (existing behavior)
            return self._start_immediate_workflow(instruction, context)
        else:
            # Scheduled execution (new behavior)
            scheduled_workflow = ScheduledWorkflow(instruction, delay, context=context)
            return self.scheduler_engine.schedule_workflow(scheduled_workflow)
```

## Benefits Analysis

### Advantages

1. **Unified Architecture**: Single system for all time-based actions
2. **Simplified Codebase**: Eliminate separate timer infrastructure
3. **Consistent API**: Same interface for immediate and delayed workflows
4. **Full Workflow Power**: Scheduled actions get planning, error handling, pause/resume
5. **Better Monitoring**: All actions tracked through workflow system
6. **Easier Testing**: Test scheduled actions like regular workflows

### Disadvantages

1. **Complexity**: Workflows are heavier than simple timers
2. **Resource Usage**: More overhead for simple "remind me" notifications
3. **Migration Effort**: Need to migrate existing timer functionality
4. **Conceptual Shift**: Users think "timers" not "scheduled workflows"

## Use Cases

### Simple Timer (Current)

```python
# Current approach:
timer = create_timer(duration=600, message="Coffee break!")

# Scheduled workflow approach:
workflow = ScheduledWorkflow(
    instruction="send notification: Coffee break!",
    delay=600,
    context={"channel": "#general", "user_id": "U123"}
)
```

### Complex Action Timer

```python
# Current approach:
timer = create_timer(duration=600, action="turn on lights")
# + separate action parsing system

# Scheduled workflow approach:
workflow = ScheduledWorkflow(
    instruction="turn on the bedroom lights",
    delay=600,
    context={"room": "bedroom", "device_context": {...}}
)
# Uses existing workflow routing automatically
```

### Recurring Actions

```python
# Scheduled workflow approach:
workflow = ScheduledWorkflow(
    instruction="check if it's raining and close windows if needed",
    delay=3600,  # 1 hour
    recurring=True,
    recurrence_pattern="hourly",
    context={"weather_check": True}
)
```

## Migration Strategy

### Phase 1: Parallel Implementation

- Keep existing timer system
- Add ScheduledWorkflow as alternative
- Test with subset of use cases

### Phase 2: Gradual Migration

- Migrate simple timers to scheduled workflows
- Update APIs to use ScheduledWorkflow
- Maintain backward compatibility

### Phase 3: Full Replacement

- Remove timer-specific infrastructure
- Unified scheduled workflow system
- Update documentation and examples

## Technical Considerations

### Storage

- **Redis Sorted Sets**: For time-based scheduling
- **Workflow Persistence**: Reuse existing TaskContext storage
- **Cleanup**: Automatic cleanup of completed scheduled workflows

### Performance

- **Polling vs Events**: Redis keyspace notifications for execution triggers
- **Batch Processing**: Execute multiple ready workflows together
- **Resource Management**: Limit concurrent scheduled workflow executions

### Error Handling

- **Retry Logic**: Failed scheduled workflows can be retried
- **Fallback Actions**: Default actions when scheduled workflow fails
- **Monitoring**: Track scheduled workflow success rates

## Future Enhancements

### Advanced Scheduling

- **Cron-like Patterns**: Complex recurring schedules
- **Conditional Execution**: Execute only if conditions met
- **Dynamic Rescheduling**: Modify scheduled workflows before execution

### Integration Opportunities

- **Calendar Integration**: Schedule workflows based on calendar events
- **Smart Home Integration**: Schedule based on sensor data
- **Weather Integration**: Schedule based on weather conditions

## Conclusion

The Scheduled Workflow Architecture represents a significant simplification and unification opportunity. By treating all time-based actions as delayed workflows, the system could eliminate the complexity of maintaining separate timer infrastructure while gaining the full power of the workflow system for scheduled actions.

This approach would be particularly powerful for complex scheduled actions that require multi-step execution, planning, or integration with multiple roles.

**Recommendation**: Consider for future major version as it would require significant architectural changes but offers substantial long-term benefits.
