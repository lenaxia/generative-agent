"""Unit tests for TaskContext core functionality.

Tests the TaskContext class which provides a high-level interface for managing
task execution, conversation history, progressive summaries, and checkpointing
functionality around the enhanced TaskGraph.
"""

import time
from unittest.mock import Mock

import pytest

from common.task_context import ExecutionState, TaskContext
from common.task_graph import TaskDependency, TaskDescription, TaskGraph, TaskStatus


class TestTaskContextUnit:
    """Unit tests for TaskContext core functionality."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample task descriptions."""
        task1 = TaskDescription(
            task_name="Task 1",
            agent_id="planning_agent",
            task_type="Planning",
            prompt="Plan the first task",
        )

        task2 = TaskDescription(
            task_name="Task 2",
            agent_id="search_agent",
            task_type="Search",
            prompt="Search for information",
        )

        return [task1, task2]

    @pytest.fixture
    def sample_dependencies(self):
        """Create sample task dependencies."""
        return [TaskDependency(source="Task 1", target="Task 2")]

    @pytest.fixture
    def task_context(self, sample_tasks, sample_dependencies):
        """Create TaskContext instance for testing."""
        return TaskContext.from_tasks(
            tasks=sample_tasks,
            dependencies=sample_dependencies,
            request_id="test_request_123",
        )

    @pytest.fixture
    def mock_task_graph(self):
        """Create mock TaskGraph."""
        graph = Mock(spec=TaskGraph)
        graph.conversation_history = []
        graph.progressive_summary = []
        graph.metadata = {}
        graph.get_ready_tasks.return_value = []
        graph.prepare_task_execution.return_value = {"prompt": "test prompt"}
        graph.mark_task_completed.return_value = []
        graph.create_checkpoint.return_value = {"graph_data": "test"}
        return graph

    def test_create_checkpoint(self, task_context):
        """Test checkpoint creation with conversation history."""
        # Add some conversation history
        task_context.add_user_message("Hello, I need help with a task")
        task_context.add_assistant_message("I'll help you with that task")
        task_context.add_system_message("Task execution started")

        # Add progressive summary
        task_context.add_summary("Started task planning phase")
        task_context.add_summary("Completed initial analysis")

        # Set some metadata
        task_context.set_metadata("priority", "high")
        task_context.set_metadata("category", "planning")

        # Start execution to set timing
        task_context.start_execution()

        # Create checkpoint
        checkpoint = task_context.create_checkpoint()

        # Verify checkpoint structure
        assert isinstance(checkpoint, dict)
        assert "context_id" in checkpoint
        assert "execution_state" in checkpoint
        assert "start_time" in checkpoint
        assert "context_version" in checkpoint

        # Verify context-specific data
        assert checkpoint["context_id"] == task_context.context_id
        assert checkpoint["execution_state"] == ExecutionState.RUNNING.value
        assert checkpoint["start_time"] == task_context.start_time
        assert checkpoint["context_version"] == "1.0"

        # Verify conversation history is included (via task graph checkpoint)
        conversation = task_context.get_conversation_history()
        assert len(conversation) == 3
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
        assert conversation[2]["role"] == "system"

        # Verify progressive summary is included
        summary = task_context.get_progressive_summary()
        assert len(summary) >= 2

    def test_restore_from_checkpoint(self, task_context):
        """Test restoration from checkpoint maintains state."""
        # Setup initial state
        task_context.add_user_message("Initial message")
        task_context.add_summary("Initial summary")
        task_context.set_metadata("test_key", "test_value")
        task_context.start_execution()

        original_context_id = task_context.context_id
        original_start_time = task_context.start_time

        # Create checkpoint
        checkpoint = task_context.create_checkpoint()

        # Create new context from checkpoint
        restored_context = TaskContext.from_checkpoint(checkpoint)

        # Verify restored state
        assert restored_context.context_id == original_context_id
        assert restored_context.execution_state == ExecutionState.RUNNING
        assert restored_context.start_time == original_start_time
        assert restored_context.context_version == "1.0"

        # Verify conversation history is restored
        restored_conversation = restored_context.get_conversation_history()
        original_conversation = task_context.get_conversation_history()
        assert len(restored_conversation) == len(original_conversation)
        assert restored_conversation[0]["content"] == "Initial message"

        # Verify progressive summary is restored
        restored_summary = restored_context.get_progressive_summary()
        original_summary = task_context.get_progressive_summary()
        assert len(restored_summary) == len(original_summary)

        # Verify metadata is restored
        assert restored_context.get_metadata("test_key") == "test_value"

    def test_progressive_summary(self, task_context):
        """Test progressive summary functionality."""
        # Add multiple summary entries
        summaries = [
            "Task planning initiated",
            "Requirements analyzed",
            "Dependencies identified",
            "Execution plan created",
            "First task completed",
        ]

        for summary in summaries:
            task_context.add_summary(summary)

        # Verify all summaries are stored
        progressive_summary = task_context.get_progressive_summary()
        assert len(progressive_summary) == 5

        # Verify summary content and structure
        for i, entry in enumerate(progressive_summary):
            assert isinstance(entry, dict)
            assert "summary" in entry
            assert "timestamp" in entry
            assert summaries[i] in entry["summary"]

        # Test summary condensation
        task_context.condense_summary(max_entries=3)
        condensed_summary = task_context.get_progressive_summary()

        # Should have 3 entries: 1 condensed + 2 most recent
        assert len(condensed_summary) == 3
        assert "[Condensed" in condensed_summary[0]["summary"]
        assert "First task completed" in condensed_summary[-1]["summary"]

    def test_conversation_history_management(self, task_context):
        """Test conversation history is properly managed."""
        # Add various message types
        task_context.add_user_message("User: Please help me plan a project")
        task_context.add_assistant_message(
            "Assistant: I'll help you create a project plan"
        )
        task_context.add_system_message("System: Task execution started")
        task_context.add_user_message("User: What's the first step?")
        task_context.add_assistant_message(
            "Assistant: Let's start by defining requirements"
        )

        # Get conversation history
        history = task_context.get_conversation_history()

        # Verify structure and content
        assert len(history) == 5

        # Verify message types and order
        expected_roles = ["user", "assistant", "system", "user", "assistant"]
        for i, entry in enumerate(history):
            assert isinstance(entry, dict)
            assert "role" in entry
            assert "content" in entry
            assert "timestamp" in entry
            assert entry["role"] == expected_roles[i]

        # Verify specific content
        assert "Please help me plan" in history[0]["content"]
        assert "I'll help you create" in history[1]["content"]
        assert "Task execution started" in history[2]["content"]
        assert "What's the first step" in history[3]["content"]
        assert "Let's start by defining" in history[4]["content"]

        # Verify timestamps are reasonable (within last few seconds)
        current_time = time.time()
        for entry in history:
            assert current_time - entry["timestamp"] < 10  # Within 10 seconds

    def test_execution_state_transitions(self, task_context):
        """Test execution state management and transitions."""
        # Initial state should be IDLE
        assert task_context.execution_state == ExecutionState.IDLE
        assert task_context.start_time is None
        assert task_context.end_time is None

        # Start execution
        start_time_before = time.time()
        task_context.start_execution()
        start_time_after = time.time()

        assert task_context.execution_state == ExecutionState.RUNNING
        assert task_context.start_time is not None
        assert start_time_before <= task_context.start_time <= start_time_after

        # Pause execution
        pause_checkpoint = task_context.pause_execution()

        assert task_context.execution_state == ExecutionState.PAUSED
        assert isinstance(pause_checkpoint, dict)
        assert pause_checkpoint["execution_state"] == ExecutionState.PAUSED.value

        # Resume execution
        task_context.resume_execution()

        assert task_context.execution_state == ExecutionState.RUNNING

        # Test resume with checkpoint
        task_context.pause_execution()
        checkpoint = task_context.create_checkpoint()

        # Modify state
        task_context.execution_state = ExecutionState.IDLE

        # Resume from checkpoint
        task_context.resume_execution(checkpoint)
        assert task_context.execution_state == ExecutionState.RUNNING

    def test_metadata_management(self, task_context):
        """Test metadata storage and retrieval."""
        # Set various metadata types
        task_context.set_metadata("priority", "high")
        task_context.set_metadata("category", "planning")
        task_context.set_metadata("estimated_duration", 3600)
        task_context.set_metadata("tags", ["urgent", "customer-facing"])
        task_context.set_metadata("config", {"retry_count": 3, "timeout": 30})

        # Test retrieval
        assert task_context.get_metadata("priority") == "high"
        assert task_context.get_metadata("category") == "planning"
        assert task_context.get_metadata("estimated_duration") == 3600
        assert task_context.get_metadata("tags") == ["urgent", "customer-facing"]
        assert task_context.get_metadata("config") == {"retry_count": 3, "timeout": 30}

        # Test default values
        assert task_context.get_metadata("nonexistent") is None
        assert (
            task_context.get_metadata("nonexistent", "default_value") == "default_value"
        )

        # Test overwriting
        task_context.set_metadata("priority", "medium")
        assert task_context.get_metadata("priority") == "medium"

    def test_task_execution_interface(self, task_context):
        """Test task execution interface methods."""
        # Get ready tasks
        ready_tasks = task_context.get_ready_tasks()
        assert isinstance(ready_tasks, list)

        # Should have at least one ready task (Task 1 has no dependencies)
        assert len(ready_tasks) >= 1

        # Get the first ready task
        first_task = ready_tasks[0]
        task_id = first_task.task_id

        # Prepare task execution
        execution_config = task_context.prepare_task_execution(task_id)
        assert isinstance(execution_config, dict)
        assert "prompt" in execution_config

        # Complete the task
        result = "Task completed successfully"
        next_tasks = task_context.complete_task(task_id, result)
        assert isinstance(next_tasks, list)

        # Verify task was marked as completed
        completed_task = task_context.task_graph.get_node_by_task_id(task_id)
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result == result

    def test_from_tasks_class_method(self, sample_tasks, sample_dependencies):
        """Test TaskContext creation from tasks and dependencies."""
        context = TaskContext.from_tasks(
            tasks=sample_tasks,
            dependencies=sample_dependencies,
            request_id="test_request_456",
            context_id="test_context_789",
        )

        # Verify context properties
        assert context.context_id == "test_context_789"
        assert context.execution_state == ExecutionState.IDLE
        assert context.context_version == "1.0"

        # Verify task graph was created correctly
        assert context.task_graph.request_id == "test_request_456"
        assert len(context.task_graph.nodes) == 2
        assert len(context.task_graph.edges) == 1

        # Verify tasks are accessible
        ready_tasks = context.get_ready_tasks()
        assert len(ready_tasks) >= 1

    def test_serialization_methods(self, task_context):
        """Test serialization and deserialization methods."""
        # Add some state
        task_context.add_user_message("Test message")
        task_context.add_summary("Test summary")
        task_context.set_metadata("test", "value")
        task_context.start_execution()

        # Test to_dict
        serialized = task_context.to_dict()
        assert isinstance(serialized, dict)
        assert "context_id" in serialized
        assert "execution_state" in serialized

        # Test from_dict (which uses from_checkpoint)
        restored = TaskContext.from_dict(serialized)
        assert restored.context_id == task_context.context_id
        assert restored.execution_state == task_context.execution_state
        assert restored.get_metadata("test") == "value"

    def test_invalid_checkpoint_handling(self):
        """Test handling of invalid checkpoint data."""
        # Test with non-dict checkpoint
        with pytest.raises(
            ValueError, match="Invalid checkpoint: must be a dictionary"
        ):
            TaskContext.from_checkpoint("invalid_checkpoint")

        with pytest.raises(
            ValueError, match="Invalid checkpoint: must be a dictionary"
        ):
            TaskContext.from_checkpoint(None)

        with pytest.raises(
            ValueError, match="Invalid checkpoint: must be a dictionary"
        ):
            TaskContext.from_checkpoint(123)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
