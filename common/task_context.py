import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from common.task_graph import (
    TaskDependency,
    TaskDescription,
    TaskGraph,
    TaskNode,
    TaskStatus,
)


class ExecutionState(str, Enum):
    """Execution state for TaskContext."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskContext:
    """
    TaskContext wrapper around enhanced TaskGraph for external state management.

    This class provides a high-level interface for managing task execution,
    conversation history, progressive summaries, and checkpointing functionality.
    """

    def __init__(self, task_graph: TaskGraph, context_id: Optional[str] = None):
        """
        Initialize TaskContext with an existing TaskGraph.

        Args:
            task_graph (TaskGraph): The underlying task graph
            context_id (str, optional): Unique identifier for this context
        """
        self.task_graph = task_graph
        self.context_id = context_id or f"ctx_{str(uuid.uuid4()).split('-')[-1]}"
        self.execution_state = ExecutionState.IDLE
        self.start_time = None
        self.end_time = None
        self.context_version = "1.0"

    @classmethod
    def from_tasks(
        cls,
        tasks: List[TaskDescription],
        dependencies: Optional[List[TaskDependency]] = None,
        request_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> "TaskContext":
        """
        Create TaskContext directly from tasks and dependencies.

        Args:
            tasks: List of task descriptions
            dependencies: List of task dependencies
            request_id: Request ID for the task graph
            context_id: Context ID for this TaskContext

        Returns:
            TaskContext: New TaskContext instance
        """
        task_graph = TaskGraph(
            tasks=tasks, dependencies=dependencies, request_id=request_id
        )
        return cls(task_graph=task_graph, context_id=context_id)

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> "TaskContext":
        """
        Create TaskContext from a checkpoint.

        Args:
            checkpoint: Checkpoint data

        Returns:
            TaskContext: Restored TaskContext instance

        Raises:
            ValueError: If checkpoint data is invalid
        """
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint: must be a dictionary")

        # Create empty task graph and restore from checkpoint
        task_graph = TaskGraph(tasks=[], dependencies=[])
        task_graph.resume_from_checkpoint(checkpoint)

        # Create context and restore context-specific state
        context = cls(task_graph=task_graph)
        context.context_id = checkpoint.get("context_id", context.context_id)
        context.execution_state = ExecutionState(
            checkpoint.get("execution_state", ExecutionState.IDLE)
        )
        context.start_time = checkpoint.get("start_time")
        context.end_time = checkpoint.get("end_time")
        context.context_version = checkpoint.get("context_version", "1.0")

        return context

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskContext":
        """
        Create TaskContext from serialized dictionary.

        Args:
            data: Serialized TaskContext data

        Returns:
            TaskContext: Restored TaskContext instance
        """
        return cls.from_checkpoint(data)

    # Conversation History Management
    def add_user_message(self, content: str):
        """Add a user message to conversation history."""
        self.task_graph.add_conversation_entry("user", content)

    def add_assistant_message(self, content: str):
        """Add an assistant message to conversation history."""
        self.task_graph.add_conversation_entry("assistant", content)

    def add_system_message(self, content: str):
        """Add a system message to conversation history."""
        self.task_graph.add_conversation_entry("system", content)

    def get_conversation_history(self) -> List[Dict]:
        """Get the complete conversation history."""
        return self.task_graph.conversation_history.copy()

    # Progressive Summary Management
    def add_summary(self, summary: str):
        """Add to the progressive summary."""
        self.task_graph.add_to_progressive_summary(summary)

    def get_progressive_summary(self) -> List[Dict]:
        """Get the progressive summary."""
        return self.task_graph.progressive_summary.copy()

    def condense_summary(self, max_entries: int = 10):
        """
        Condense the progressive summary to keep only the most recent entries.

        Args:
            max_entries: Maximum number of summary entries to keep
        """
        if len(self.task_graph.progressive_summary) > max_entries:
            # Keep the most recent entries (subtract 1 to account for condensed entry)
            recent_summaries = self.task_graph.progressive_summary[-(max_entries - 1) :]

            # Create a condensed summary of older entries
            older_count = len(self.task_graph.progressive_summary) - (max_entries - 1)
            condensed_entry = {
                "summary": f"[Condensed {older_count} earlier summary entries]",
                "timestamp": time.time(),
            }

            self.task_graph.progressive_summary = [condensed_entry] + recent_summaries

    # Metadata Management
    def set_metadata(self, key: str, value: Any):
        """Set metadata for the task context."""
        self.task_graph.set_metadata(key, value)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.task_graph.get_metadata(key, default)

    # Checkpoint Management
    def create_checkpoint(self) -> Dict:
        """
        Create a comprehensive checkpoint of the current state.

        Returns:
            Dict: Checkpoint containing all context state
        """
        # Get base checkpoint from task graph
        checkpoint = self.task_graph.create_checkpoint()

        # Add TaskContext-specific state
        checkpoint.update(
            {
                "context_id": self.context_id,
                "execution_state": self.execution_state.value,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "context_version": self.context_version,
            }
        )

        return checkpoint

    # Serialization
    def to_dict(self) -> Dict:
        """
        Serialize TaskContext to dictionary.

        Returns:
            Dict: Serialized TaskContext data
        """
        return self.create_checkpoint()

    # Task Execution Interface
    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks that are ready to execute."""
        return self.task_graph.get_ready_tasks()

    def prepare_task_execution(self, task_id: str) -> Dict:
        """
        Prepare configuration for task execution.

        Args:
            task_id: Task ID to prepare execution for

        Returns:
            Dict: Execution configuration
        """
        return self.task_graph.prepare_task_execution(task_id)

    def complete_task(self, task_id: str, result: str) -> List[TaskNode]:
        """
        Mark a task as completed and return next ready tasks.

        Args:
            task_id: Task ID to complete
            result: Task result

        Returns:
            List[TaskNode]: Next ready tasks
        """
        return self.task_graph.mark_task_completed(task_id, result)

    # Execution State Management
    def start_execution(self):
        """Start task execution."""
        self.execution_state = ExecutionState.RUNNING
        self.start_time = time.time()

    def pause_execution(self) -> Dict:
        """
        Pause execution and return checkpoint.

        Returns:
            Dict: Checkpoint for resuming execution
        """
        self.execution_state = ExecutionState.PAUSED
        return self.create_checkpoint()

    def resume_execution(self, checkpoint: Optional[Dict] = None):
        """
        Resume execution from checkpoint or current state.

        Args:
            checkpoint: Optional checkpoint to resume from
        """
        if checkpoint:
            # Restore state from checkpoint
            restored_context = self.from_checkpoint(checkpoint)
            self.task_graph = restored_context.task_graph
            self.context_id = restored_context.context_id
            self.start_time = restored_context.start_time
            self.end_time = restored_context.end_time

        self.execution_state = ExecutionState.RUNNING

    def complete_execution(self):
        """Mark execution as completed."""
        self.execution_state = ExecutionState.COMPLETED
        self.end_time = time.time()

    def fail_execution(self, error: str):
        """Mark execution as failed."""
        self.execution_state = ExecutionState.FAILED
        self.end_time = time.time()
        self.set_metadata("error", error)

    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.execution_state == ExecutionState.RUNNING

    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.task_graph.is_complete()

    # Performance Metrics
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the task execution.

        Returns:
            Dict: Performance metrics
        """
        total_tasks = len(self.task_graph.nodes)
        completed_tasks = len(
            [
                n
                for n in self.task_graph.nodes.values()
                if n.status == TaskStatus.COMPLETED
            ]
        )
        failed_tasks = len(self.task_graph.get_failed_tasks())

        execution_time = None
        if self.start_time:
            end_time = self.end_time or time.time()
            execution_time = end_time - self.start_time

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": total_tasks - completed_tasks - failed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "execution_time": execution_time,
            "execution_state": self.execution_state.value,
            "context_id": self.context_id,
        }

    def __str__(self) -> str:
        """String representation of TaskContext."""
        metrics = self.get_performance_metrics()
        return (
            f"TaskContext(id={self.context_id}, "
            f"state={self.execution_state.value}, "
            f"tasks={metrics['completed_tasks']}/{metrics['total_tasks']})"
        )

    def __repr__(self) -> str:
        """Detailed representation of TaskContext."""
        return self.__str__()
