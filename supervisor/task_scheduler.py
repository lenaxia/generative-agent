import logging
import time
import asyncio
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import heapq
import json

from supervisor.request_manager import RequestManager
from common.task_context import TaskContext, ExecutionState
from common.task_graph import TaskNode, TaskStatus
from common.message_bus import MessageBus, MessageType

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SchedulerState(str, Enum):
    """Scheduler execution states."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


@dataclass
class QueuedTask:
    """Represents a task in the scheduler queue."""
    priority: TaskPriority
    scheduled_time: float
    task: TaskNode
    context: TaskContext
    task_id: str = field(init=False)
    
    def __post_init__(self):
        self.task_id = self.task.task_id
    
    def __lt__(self, other):
        """Priority queue ordering: higher priority first, then by scheduled time."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.scheduled_time < other.scheduled_time  # Earlier scheduled time first


class TaskScheduler:
    """
    Advanced task scheduler with priority queuing, concurrency control, and pause/resume.
    
    This complements the existing Heartbeat mechanism by providing detailed task
    execution management while Heartbeat handles system-wide monitoring.
    """
    
    def __init__(self, request_manager: RequestManager, message_bus: MessageBus,
                 max_concurrent_tasks: int = 5, checkpoint_interval: int = 300):
        """
        Initialize TaskScheduler.
        
        Args:
            request_manager: RequestManager for task delegation
            message_bus: Message bus for communication
            max_concurrent_tasks: Maximum concurrent task execution
            checkpoint_interval: Checkpoint creation interval in seconds
        """
        self.request_manager = request_manager
        self.message_bus = message_bus
        self.max_concurrent_tasks = max_concurrent_tasks
        self.checkpoint_interval = checkpoint_interval
        
        # Scheduler state
        self.state = SchedulerState.IDLE
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.last_checkpoint_time: Optional[float] = None
        
        # Task management
        self.task_queue: List[QueuedTask] = []
        self.running_tasks: Dict[str, Dict] = {}
        
        # Subscribe to task completion and error events
        self.message_bus.subscribe(self, MessageType.TASK_RESPONSE, self.handle_task_completion)
        self.message_bus.subscribe(self, MessageType.AGENT_ERROR, self.handle_task_error)
        
        logger.info(f"TaskScheduler initialized with max_concurrent_tasks={max_concurrent_tasks}")
    
    def start(self):
        """Start the task scheduler."""
        self.state = SchedulerState.RUNNING
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        logger.info("TaskScheduler started")
    
    def stop(self):
        """Stop the task scheduler."""
        self.state = SchedulerState.STOPPED
        self.end_time = time.time()
        logger.info("TaskScheduler stopped")
    
    def pause(self) -> Dict:
        """
        Pause the scheduler and create checkpoint.
        
        Returns:
            Dict: Checkpoint data for resuming
        """
        self.state = SchedulerState.PAUSED
        
        checkpoint = {
            "scheduler_state": {
                "state": self.state.value,
                "start_time": self.start_time,
                "last_checkpoint_time": self.last_checkpoint_time,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "checkpoint_interval": self.checkpoint_interval
            },
            "task_queue": [
                {
                    "priority": task.priority.value,
                    "scheduled_time": task.scheduled_time,
                    "task_id": task.task_id,
                    "context_id": task.context.context_id
                }
                for task in self.task_queue
            ],
            "running_tasks": {
                task_id: {
                    "start_time": info["start_time"],
                    "context_id": info["context"].context_id
                }
                for task_id, info in self.running_tasks.items()
            },
            "timestamp": time.time()
        }
        
        logger.info("TaskScheduler paused and checkpoint created")
        return checkpoint
    
    def resume(self, checkpoint: Optional[Dict] = None) -> bool:
        """
        Resume the scheduler from checkpoint.
        
        Args:
            checkpoint: Optional checkpoint to resume from
            
        Returns:
            bool: True if resumed successfully
        """
        try:
            if checkpoint:
                # Restore scheduler state
                scheduler_state = checkpoint.get("scheduler_state", {})
                self.start_time = scheduler_state.get("start_time", time.time())
                self.last_checkpoint_time = scheduler_state.get("last_checkpoint_time", time.time())
                
                # Note: Task queue and running tasks would need to be restored
                # from their respective contexts - simplified for now
                logger.info("TaskScheduler resumed from checkpoint")
            else:
                logger.info("TaskScheduler resumed without checkpoint")
            
            self.state = SchedulerState.RUNNING
            return True
            
        except Exception as e:
            logger.error(f"Error resuming TaskScheduler: {e}")
            return False
    
    def schedule_task(self, context: TaskContext, task: TaskNode, 
                     priority: TaskPriority = TaskPriority.NORMAL):
        """
        Schedule a task for execution.
        
        Args:
            context: Task context
            task: Task node to schedule
            priority: Task priority level
        """
        queued_task = QueuedTask(
            priority=priority,
            scheduled_time=time.time(),
            task=task,
            context=context
        )
        
        # Add to priority queue (heapq maintains order)
        heapq.heappush(self.task_queue, queued_task)
        
        logger.info(f"Task '{task.task_id}' scheduled with priority {priority.name}")
        
        # Process queue if scheduler is running
        if self.state == SchedulerState.RUNNING:
            self._process_task_queue()
    
    def _process_task_queue(self):
        """Process the task queue respecting concurrency limits."""
        while (len(self.running_tasks) < self.max_concurrent_tasks and 
               self.task_queue and 
               self.state == SchedulerState.RUNNING):
            
            # Get highest priority task
            queued_task = heapq.heappop(self.task_queue)
            
            # Start task execution
            self._start_task_execution(queued_task)
    
    def _start_task_execution(self, queued_task: QueuedTask):
        """
        Start execution of a queued task.
        
        Args:
            queued_task: The task to start executing
        """
        task = queued_task.task
        context = queued_task.context
        
        # Add to running tasks
        self.running_tasks[task.task_id] = {
            "task": task,
            "context": context,
            "start_time": time.time(),
            "priority": queued_task.priority
        }
        
        # Delegate to RequestManager
        try:
            self.request_manager.delegate_task(context, task)
            logger.info(f"Started execution of task '{task.task_id}' (priority: {queued_task.priority.name})")
        except Exception as e:
            logger.error(f"Error starting task '{task.task_id}': {e}")
            # Remove from running tasks on error
            del self.running_tasks[task.task_id]
    
    def handle_task_completion(self, completion_data: Dict):
        """
        Handle task completion events.
        
        Args:
            completion_data: Task completion information
        """
        task_id = completion_data.get("task_id")
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            execution_time = time.time() - task_info["start_time"]
            
            logger.info(f"Task '{task_id}' completed in {execution_time:.2f}s")
            
            # Process more tasks from queue
            self._process_task_queue()
            
            # Create checkpoint if interval elapsed
            self._maybe_create_checkpoint()
    
    def handle_task_error(self, error_data: Dict):
        """
        Handle task error events.
        
        Args:
            error_data: Task error information
        """
        task_id = error_data.get("task_id")
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            error_message = error_data.get("error_message", "Unknown error")
            
            logger.error(f"Task '{task_id}' failed: {error_message}")
            
            # Process more tasks from queue
            self._process_task_queue()
    
    def _maybe_create_checkpoint(self):
        """Create checkpoint if interval has elapsed."""
        current_time = time.time()
        if (self.last_checkpoint_time and 
            current_time - self.last_checkpoint_time >= self.checkpoint_interval):
            
            checkpoint = self.pause()
            # In a full implementation, this would be persisted
            logger.info("Automatic checkpoint created")
            self.state = SchedulerState.RUNNING  # Resume after checkpoint
            self.last_checkpoint_time = current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get scheduler performance metrics.
        
        Returns:
            Dict: Scheduler metrics
        """
        uptime = None
        if self.start_time:
            end_time = self.end_time or time.time()
            uptime = end_time - self.start_time
        
        return {
            "state": self.state,
            "uptime": uptime,
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "checkpoint_interval": self.checkpoint_interval,
            "last_checkpoint_time": self.last_checkpoint_time,
            "queue_priorities": [task.priority.name for task in self.task_queue],
            "running_task_priorities": [
                info["priority"].name for info in self.running_tasks.values()
            ]
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get detailed queue status.
        
        Returns:
            Dict: Queue status information
        """
        return {
            "total_queued": len(self.task_queue),
            "total_running": len(self.running_tasks),
            "available_slots": self.max_concurrent_tasks - len(self.running_tasks),
            "queue_by_priority": {
                priority.name: len([t for t in self.task_queue if t.priority == priority])
                for priority in TaskPriority
            },
            "running_by_priority": {
                priority.name: len([info for info in self.running_tasks.values() 
                                  if info["priority"] == priority])
                for priority in TaskPriority
            }
        }
    
    def clear_queue(self):
        """Clear all queued tasks (running tasks continue)."""
        cleared_count = len(self.task_queue)
        self.task_queue.clear()
        logger.info(f"Cleared {cleared_count} queued tasks")
    
    def get_running_task_ids(self) -> List[str]:
        """Get list of currently running task IDs."""
        return list(self.running_tasks.keys())
    
    def get_queued_task_ids(self) -> List[str]:
        """Get list of queued task IDs."""
        return [task.task_id for task in self.task_queue]
    
    def __str__(self) -> str:
        """String representation of TaskScheduler."""
        return (f"TaskScheduler(state={self.state.value}, "
                f"queued={len(self.task_queue)}, "
                f"running={len(self.running_tasks)}/{self.max_concurrent_tasks})")