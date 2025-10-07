import json
import unittest

from common.task_graph import TaskDependency, TaskDescription, TaskGraph, TaskStatus


class TestTaskGraphCheckpoints(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with a simple task graph."""
        self.tasks = [
            TaskDescription(
                task_name="task1",
                agent_id="planning_agent",
                task_type="planning",
                prompt="Create a plan for the user request",
            ),
            TaskDescription(
                task_name="task2",
                agent_id="search_agent",
                task_type="search",
                prompt="Search for information based on the plan",
            ),
        ]

        self.dependencies = [
            TaskDependency(source="task1", target="task2", condition=None)
        ]

        self.task_graph = TaskGraph(
            tasks=self.tasks,
            dependencies=self.dependencies,
            request_id="test_request_123",
        )

    def test_create_checkpoint_empty_graph(self):
        """Test creating a checkpoint with no completed tasks."""
        checkpoint = self.task_graph.create_checkpoint()

        assert checkpoint is not None
        assert "timestamp" in checkpoint
        assert "task_graph_state" in checkpoint
        assert "conversation_history" in checkpoint
        assert "progressive_summary" in checkpoint
        assert "metadata" in checkpoint

        # Verify task states are preserved
        task_states = checkpoint["task_graph_state"]["nodes"]
        assert len(task_states) == 2

        # All tasks should be PENDING initially
        for task_state in task_states:
            assert task_state["status"] == TaskStatus.PENDING.value

    def test_create_checkpoint_with_completed_tasks(self):
        """Test creating a checkpoint after some tasks are completed."""
        # Complete first task
        task1_id = self.task_graph.task_name_map["task1"]
        self.task_graph.mark_task_completed(task1_id, "Task 1 completed successfully")

        checkpoint = self.task_graph.create_checkpoint()

        # Verify checkpoint contains completed task state
        task_states = checkpoint["task_graph_state"]["nodes"]
        task1_state = next(t for t in task_states if t["task_name"] == "task1")
        task2_state = next(t for t in task_states if t["task_name"] == "task2")

        assert task1_state["status"] == TaskStatus.COMPLETED.value
        assert task2_state["status"] == TaskStatus.PENDING.value

        # Verify history is captured
        assert len(checkpoint["task_graph_state"]["history"]) > 0

    def test_create_checkpoint_serializable(self):
        """Test that checkpoints can be serialized to JSON."""
        checkpoint = self.task_graph.create_checkpoint()

        # Should not raise an exception
        json_str = json.dumps(checkpoint)
        assert isinstance(json_str, str)

        # Should be able to deserialize back
        deserialized = json.loads(json_str)
        assert checkpoint["task_graph_state"] == deserialized["task_graph_state"]

    def test_resume_from_checkpoint_basic(self):
        """Test basic checkpoint resume functionality."""
        # Create initial checkpoint
        self.task_graph.create_checkpoint()

        # Complete a task
        task1_id = self.task_graph.task_name_map["task1"]
        self.task_graph.mark_task_completed(task1_id, "Task 1 result")

        # Create checkpoint after completion
        checkpoint_with_progress = self.task_graph.create_checkpoint()

        # Create new task graph and resume from checkpoint
        new_task_graph = TaskGraph(tasks=[], dependencies=[])
        new_task_graph.resume_from_checkpoint(checkpoint_with_progress)

        # Verify state was restored
        assert len(new_task_graph.nodes) == 2
        restored_task1 = new_task_graph.get_node_by_task_id(task1_id)
        assert restored_task1 is not None
        assert restored_task1.status == TaskStatus.COMPLETED
        assert restored_task1.result == "Task 1 result"

    def test_resume_from_checkpoint_preserves_dependencies(self):
        """Test that resuming from checkpoint preserves task dependencies."""
        # Complete first task and create checkpoint
        task1_id = self.task_graph.task_name_map["task1"]
        self.task_graph.mark_task_completed(task1_id, "Task 1 result")
        checkpoint = self.task_graph.create_checkpoint()

        # Resume from checkpoint
        new_task_graph = TaskGraph(tasks=[], dependencies=[])
        new_task_graph.resume_from_checkpoint(checkpoint)

        # Verify dependencies are preserved
        assert len(new_task_graph.edges) == 1

        task2_id = new_task_graph.task_name_map["task2"]
        task2_node = new_task_graph.get_node_by_task_id(task2_id)
        assert len(task2_node.inbound_edges) == 1
        assert task2_node.inbound_edges[0].source_id == task1_id

    def test_resume_from_checkpoint_invalid_data(self):
        """Test error handling for invalid checkpoint data."""
        invalid_checkpoint = {"invalid": "data"}

        new_task_graph = TaskGraph(tasks=[], dependencies=[])

        with self.assertRaises(ValueError):
            new_task_graph.resume_from_checkpoint(invalid_checkpoint)

    def test_checkpoint_progressive_summary_tracking(self):
        """Test that checkpoints track progressive summaries."""
        # Add some summary data
        self.task_graph.add_to_progressive_summary("Initial planning phase completed")

        checkpoint = self.task_graph.create_checkpoint()

        assert "progressive_summary" in checkpoint
        summary_data = checkpoint["progressive_summary"]
        assert "Initial planning phase completed" in str(summary_data)

    def test_checkpoint_conversation_history_tracking(self):
        """Test that checkpoints track conversation history."""
        # Add conversation history
        self.task_graph.add_conversation_entry("user", "Please create a plan")
        self.task_graph.add_conversation_entry(
            "assistant", "I'll create a plan for you"
        )

        checkpoint = self.task_graph.create_checkpoint()

        assert "conversation_history" in checkpoint
        history = checkpoint["conversation_history"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_multiple_checkpoints_chronological(self):
        """Test that multiple checkpoints maintain chronological order."""
        checkpoint1 = self.task_graph.create_checkpoint()
        # time.sleep(0.01)  # Removed for faster tests - use counter instead

        task1_id = self.task_graph.task_name_map["task1"]
        self.task_graph.mark_task_completed(task1_id, "Task 1 result")

        checkpoint2 = self.task_graph.create_checkpoint()

        assert checkpoint1["timestamp"] < checkpoint2["timestamp"]

    def test_checkpoint_metadata_preservation(self):
        """Test that checkpoint metadata is preserved."""
        # Add metadata
        self.task_graph.set_metadata("user_id", "user123")
        self.task_graph.set_metadata("session_id", "session456")

        checkpoint = self.task_graph.create_checkpoint()

        metadata = checkpoint["metadata"]
        assert metadata["user_id"] == "user123"
        assert metadata["session_id"] == "session456"


if __name__ == "__main__":
    unittest.main()
