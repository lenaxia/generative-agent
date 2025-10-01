# TaskScheduler Implementation Guide

## Overview

The TaskScheduler is a sophisticated task management system that complements the existing Heartbeat mechanism by providing advanced task queue management, priority scheduling, and pause/resume functionality. It works seamlessly with the refactored RequestManager and Universal Agent system.

## Architecture Relationship

### **Heartbeat vs TaskScheduler**

| Component | Purpose | Scope | Pattern |
|-----------|---------|-------|---------|
| **Heartbeat** | System health monitoring | Macro-level (system-wide) | Timer-based loop (5s intervals) |
| **TaskScheduler** | Task execution management | Micro-level (individual tasks) | Event-driven with message bus |

### **Complementary Design**
- **Heartbeat**: Monitors overall system health and can use TaskScheduler for `schedule_new_tasks()`
- **TaskScheduler**: Manages individual task execution with priorities, concurrency, and state management

## Key Features

### ✅ **Priority-Based Task Queuing**
```python
class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2  
    HIGH = 3
    CRITICAL = 4
```

Tasks are automatically ordered by priority using a heap queue, ensuring critical tasks execute first.

### ✅ **Concurrency Control**
- Configurable maximum concurrent tasks (default: 5)
- Automatic queue processing when slots become available
- Load balancing across available execution slots

### ✅ **Pause/Resume with Checkpoints**
- Full scheduler state serialization
- Task queue and running task preservation
- Automatic checkpoint creation at configurable intervals

### ✅ **Message Bus Integration**
- Subscribes to `TASK_RESPONSE` and `AGENT_ERROR` events
- Publishes task lifecycle events
- Seamless integration with existing communication patterns

### ✅ **Performance Monitoring**
- Detailed metrics on queue status, running tasks, and performance
- Priority distribution tracking
- Execution time monitoring

## Implementation Details

### **Core Classes**

#### **QueuedTask Dataclass**
```python
@dataclass
class QueuedTask:
    priority: TaskPriority
    scheduled_time: float
    task: TaskNode
    context: TaskContext
    task_id: str = field(init=False)
    
    def __lt__(self, other):
        # Higher priority first, then by scheduled time
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.scheduled_time < other.scheduled_time
```

#### **TaskScheduler Class**
```python
class TaskScheduler:
    def __init__(self, request_manager: RequestManager, message_bus: MessageBus,
                 max_concurrent_tasks: int = 5, checkpoint_interval: int = 300)
```

### **Key Methods**

#### **Task Management**
```python
def schedule_task(self, context: TaskContext, task: TaskNode, 
                 priority: TaskPriority = TaskPriority.NORMAL)
def _process_task_queue()  # Respects concurrency limits
def _start_task_execution(self, queued_task: QueuedTask)
```

#### **Lifecycle Control**
```python
def start()  # Start scheduler
def stop()   # Stop scheduler
def pause() -> Dict  # Pause and create checkpoint
def resume(checkpoint: Optional[Dict] = None) -> bool  # Resume from checkpoint
```

#### **Event Handlers**
```python
def handle_task_completion(self, completion_data: Dict)
def handle_task_error(self, error_data: Dict)
```

#### **Monitoring**
```python
def get_metrics() -> Dict[str, Any]
def get_queue_status() -> Dict[str, Any]
def get_running_task_ids() -> List[str]
def get_queued_task_ids() -> List[str]
```

## Usage Examples

### **Basic Usage**
```python
from supervisor.task_scheduler import TaskScheduler, TaskPriority
from supervisor.request_manager import RequestManager
from common.message_bus import MessageBus

# Initialize components
scheduler = TaskScheduler(request_manager, message_bus)

# Start scheduler
scheduler.start()

# Schedule tasks with priorities
scheduler.schedule_task(context, urgent_task, TaskPriority.HIGH)
scheduler.schedule_task(context, normal_task, TaskPriority.NORMAL)
scheduler.schedule_task(context, background_task, TaskPriority.LOW)
```

### **Pause/Resume Operations**
```python
# Pause scheduler and get checkpoint
checkpoint = scheduler.pause()

# Later, resume from checkpoint
success = scheduler.resume(checkpoint)
if success:
    print("Scheduler resumed successfully")
```

### **Monitoring**
```python
# Get detailed metrics
metrics = scheduler.get_metrics()
print(f"State: {metrics['state']}")
print(f"Queued: {metrics['queued_tasks']}")
print(f"Running: {metrics['running_tasks']}/{metrics['max_concurrent_tasks']}")

# Get queue status
queue_status = scheduler.get_queue_status()
print(f"Available slots: {queue_status['available_slots']}")
print(f"Priority distribution: {queue_status['queue_by_priority']}")
```

### **Integration with Heartbeat**
```python
# Enhanced Heartbeat using TaskScheduler
class EnhancedHeartbeat(Heartbeat):
    def __init__(self, supervisor, task_scheduler, interval=5):
        super().__init__(supervisor, interval)
        self.task_scheduler = task_scheduler
    
    def schedule_new_tasks(self):
        """Enhanced scheduling using TaskScheduler."""
        # Get pending tasks from supervisor
        pending_tasks = self.supervisor.get_pending_tasks()
        
        for task_info in pending_tasks:
            priority = self._determine_priority(task_info)
            self.task_scheduler.schedule_task(
                task_info["context"], 
                task_info["task"], 
                priority
            )
```

## Configuration Options

### **Scheduler Configuration**
```python
TaskScheduler(
    request_manager=request_manager,
    message_bus=message_bus,
    max_concurrent_tasks=5,      # Concurrency limit
    checkpoint_interval=300      # Checkpoint every 5 minutes
)
```

### **Priority Mapping**
The scheduler automatically maps task types to appropriate priorities:
```python
# Example priority assignment logic
def determine_task_priority(task: TaskNode) -> TaskPriority:
    if task.agent_id == "planning_agent":
        return TaskPriority.HIGH      # Planning is critical
    elif task.agent_id in ["search_agent", "weather_agent"]:
        return TaskPriority.LOW       # Simple lookups
    else:
        return TaskPriority.NORMAL    # Default priority
```

## Performance Characteristics

### **Scalability**
- **Queue Management**: O(log n) insertion and removal using heapq
- **Concurrency**: Configurable limits prevent resource exhaustion
- **Memory**: Efficient dataclass-based task representation

### **Reliability**
- **Checkpointing**: Automatic state preservation
- **Error Handling**: Graceful task failure recovery
- **Message Bus**: Reliable event-driven communication

### **Monitoring**
- **Real-time Metrics**: Queue status, execution times, priority distribution
- **Health Checks**: Running task monitoring and timeout detection
- **Performance Tracking**: Throughput and latency measurements

## Integration Points

### **With RequestManager**
```python
# TaskScheduler delegates to RequestManager
self.request_manager.delegate_task(context, task)
```

### **With TaskContext**
```python
# Uses TaskContext for state management
context.prepare_task_execution(task_id)
context.complete_task(task_id, result)
```

### **With Message Bus**
```python
# Subscribes to task lifecycle events
self.message_bus.subscribe(self, MessageType.TASK_RESPONSE, self.handle_task_completion)
self.message_bus.subscribe(self, MessageType.AGENT_ERROR, self.handle_task_error)
```

## Testing Coverage

### **Comprehensive Test Suite**
- ✅ Initialization and configuration
- ✅ Priority-based task queuing
- ✅ Concurrency limit enforcement
- ✅ Pause/resume functionality
- ✅ Error handling and recovery
- ✅ Message bus integration
- ✅ Performance metrics
- ✅ State management

### **Test Results**
```
13 passed, 2 warnings in 0.25s
```

All tests pass, demonstrating robust functionality across all features.

## Migration Benefits

### **Enhanced Capabilities**
- **Priority Scheduling**: Critical tasks execute first
- **Concurrency Control**: Optimal resource utilization
- **Pause/Resume**: Long-running workflows can be interrupted and resumed
- **Advanced Monitoring**: Detailed performance and queue metrics

### **Backward Compatibility**
- **Heartbeat Integration**: Complements existing monitoring
- **Message Bus**: Uses existing communication patterns
- **RequestManager**: Works with refactored RequestManager

### **Performance Improvements**
- **Efficient Queuing**: O(log n) priority queue operations
- **Resource Management**: Prevents system overload
- **State Persistence**: Reliable checkpoint/resume functionality

## Next Steps

### **Phase 4: MCP Integration**
The TaskScheduler is ready for MCP server integration:
- Task scheduling can incorporate MCP tool availability
- Priority assignment can consider MCP server load
- Checkpoint system can preserve MCP connection state

### **Enhanced Heartbeat Integration**
```python
# Future enhancement: Heartbeat uses TaskScheduler
def schedule_new_tasks(self):
    """Use TaskScheduler for advanced task management."""
    self.task_scheduler.schedule_task(context, task, priority)
```

## Conclusion

The TaskScheduler successfully complements the existing Heartbeat mechanism while providing advanced task management capabilities. It integrates seamlessly with the Universal Agent system and provides the foundation for sophisticated workflow management with pause/resume functionality.

The implementation follows StrandsAgent patterns and maintains compatibility with the existing architecture while adding powerful new capabilities for task orchestration and management.