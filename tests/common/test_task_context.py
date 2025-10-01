import unittest
import time
import json
from unittest.mock import Mock, patch
from common.task_graph import TaskGraph, TaskDescription, TaskStatus, TaskNode, TaskDependency


class TaskContext:
    """
    TaskContext wrapper around enhanced TaskGraph for external state management.
    This is the class we're testing - it will be implemented after tests are written.
    """
    pass


class TestTaskContext(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for TaskContext."""
        self.tasks = [
            TaskDescription(
                task_name="planning_task",
                agent_id="planning_agent",
                task_type="planning",
                prompt="Create a plan for: {input}"
            ),
            TaskDescription(
                task_name="execution_task", 
                agent_id="execution_agent",
                task_type="execution",
                prompt="Execute the plan: {input}"
            )
        ]
        
        self.dependencies = [
            TaskDependency(source="planning_task", target="execution_task", condition=None)
        ]
        
        # Create underlying TaskGraph
        self.task_graph = TaskGraph(
            tasks=self.tasks,
            dependencies=self.dependencies,
            request_id="test_context_request"
        )

    def test_task_context_initialization_with_task_graph(self):
        """Test TaskContext can be initialized with an existing TaskGraph."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        self.assertIsNotNone(context.task_graph)
        self.assertEqual(context.task_graph.request_id, "test_context_request")
        self.assertEqual(len(context.task_graph.nodes), 2)

    def test_task_context_initialization_from_tasks(self):
        """Test TaskContext can be initialized directly from tasks and dependencies."""
        from common.task_context import TaskContext
        
        context = TaskContext.from_tasks(
            tasks=self.tasks,
            dependencies=self.dependencies,
            request_id="direct_init_test"
        )
        
        self.assertIsNotNone(context.task_graph)
        self.assertEqual(context.task_graph.request_id, "direct_init_test")
        self.assertEqual(len(context.task_graph.nodes), 2)

    def test_task_context_conversation_history_management(self):
        """Test TaskContext provides conversation history management."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Add conversation entries
        context.add_user_message("I need help with a project")
        context.add_assistant_message("I'll help you create a plan")
        context.add_system_message("Task planning initiated")
        
        history = context.get_conversation_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[1]['role'], 'assistant')
        self.assertEqual(history[2]['role'], 'system')

    def test_task_context_progressive_summary_management(self):
        """Test TaskContext manages progressive summaries."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Add progressive summaries
        context.add_summary("User requested project assistance")
        context.add_summary("Planning phase initiated")
        context.add_summary("Initial requirements gathered")
        
        summaries = context.get_progressive_summary()
        self.assertEqual(len(summaries), 3)
        
        # Test summary condensation
        context.condense_summary(max_entries=2)
        condensed = context.get_progressive_summary()
        self.assertLessEqual(len(condensed), 2)

    def test_task_context_metadata_management(self):
        """Test TaskContext provides metadata management."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Set various metadata
        context.set_metadata("user_id", "user123")
        context.set_metadata("session_id", "session456")
        context.set_metadata("priority", "high")
        
        self.assertEqual(context.get_metadata("user_id"), "user123")
        self.assertEqual(context.get_metadata("session_id"), "session456")
        self.assertEqual(context.get_metadata("priority"), "high")
        self.assertIsNone(context.get_metadata("nonexistent"))
        self.assertEqual(context.get_metadata("nonexistent", "default"), "default")

    def test_task_context_checkpoint_creation(self):
        """Test TaskContext can create comprehensive checkpoints."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Add some state
        context.add_user_message("Test message")
        context.add_summary("Test summary")
        context.set_metadata("test_key", "test_value")
        
        checkpoint = context.create_checkpoint()
        
        self.assertIn('timestamp', checkpoint)
        self.assertIn('task_graph_state', checkpoint)
        self.assertIn('conversation_history', checkpoint)
        self.assertIn('progressive_summary', checkpoint)
        self.assertIn('metadata', checkpoint)
        self.assertIn('context_version', checkpoint)

    def test_task_context_checkpoint_resume(self):
        """Test TaskContext can resume from checkpoints."""
        from common.task_context import TaskContext
        
        # Create original context with state
        original_context = TaskContext(task_graph=self.task_graph)
        original_context.add_user_message("Original message")
        original_context.add_summary("Original summary")
        original_context.set_metadata("original_key", "original_value")
        
        # Complete a task
        task_id = original_context.task_graph.task_name_map["planning_task"]
        original_context.task_graph.mark_task_completed(task_id, "Planning completed")
        
        # Create checkpoint
        checkpoint = original_context.create_checkpoint()
        
        # Resume from checkpoint
        new_context = TaskContext.from_checkpoint(checkpoint)
        
        # Verify state was restored
        self.assertEqual(len(new_context.get_conversation_history()), 1)
        self.assertEqual(new_context.get_metadata("original_key"), "original_value")
        
        # Verify task state was restored
        restored_node = new_context.task_graph.get_node_by_task_id(task_id)
        self.assertEqual(restored_node.status, TaskStatus.COMPLETED)

    def test_task_context_serialization(self):
        """Test TaskContext can be serialized for persistence."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        context.add_user_message("Test message")
        context.set_metadata("test", "value")
        
        # Serialize to dict
        serialized = context.to_dict()
        self.assertIsInstance(serialized, dict)
        
        # Should be JSON serializable
        json_str = json.dumps(serialized)
        self.assertIsInstance(json_str, str)
        
        # Deserialize back
        deserialized_dict = json.loads(json_str)
        restored_context = TaskContext.from_dict(deserialized_dict)
        
        self.assertEqual(len(restored_context.get_conversation_history()), 1)
        self.assertEqual(restored_context.get_metadata("test"), "value")

    def test_task_context_task_delegation(self):
        """Test TaskContext provides task delegation interface."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Get ready tasks
        ready_tasks = context.get_ready_tasks()
        self.assertGreater(len(ready_tasks), 0)
        
        # Get execution config for a task
        task_id = ready_tasks[0].task_id
        exec_config = context.prepare_task_execution(task_id)
        
        self.assertIn('role', exec_config)
        self.assertIn('llm_type', exec_config)
        self.assertIn('tools', exec_config)
        self.assertIn('context', exec_config)

    def test_task_context_completion_tracking(self):
        """Test TaskContext tracks task completion and updates state."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Complete first task
        task_id = context.task_graph.task_name_map["planning_task"]
        result = {"plan": "Step 1, Step 2, Step 3"}
        
        next_tasks = context.complete_task(task_id, json.dumps(result))
        
        # Should return next ready tasks
        self.assertIsInstance(next_tasks, list)
        
        # Progressive summary should be updated
        summaries = context.get_progressive_summary()
        self.assertGreater(len(summaries), 0)

    def test_task_context_pause_resume_functionality(self):
        """Test TaskContext supports pause/resume functionality."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Start execution
        context.start_execution()
        self.assertTrue(context.is_running())
        
        # Pause execution
        pause_checkpoint = context.pause_execution()
        self.assertFalse(context.is_running())
        self.assertIsNotNone(pause_checkpoint)
        
        # Resume execution
        context.resume_execution(pause_checkpoint)
        self.assertTrue(context.is_running())

    def test_task_context_error_handling(self):
        """Test TaskContext handles errors gracefully."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Test invalid task ID
        with self.assertRaises(ValueError):
            context.prepare_task_execution("invalid_task_id")
        
        # Test invalid checkpoint
        with self.assertRaises(ValueError):
            TaskContext.from_checkpoint({"invalid": "checkpoint"})

    def test_task_context_state_isolation(self):
        """Test that TaskContext instances have isolated state."""
        from common.task_context import TaskContext
        
        # Create separate TaskGraphs for proper isolation
        task_graph1 = TaskGraph(
            tasks=self.tasks,
            dependencies=self.dependencies,
            request_id="isolation_test_1"
        )
        task_graph2 = TaskGraph(
            tasks=self.tasks,
            dependencies=self.dependencies,
            request_id="isolation_test_2"
        )
        
        context1 = TaskContext(task_graph=task_graph1)
        context2 = TaskContext(task_graph=task_graph2)
        
        # Modify context1
        context1.add_user_message("Context 1 message")
        context1.set_metadata("context", "1")
        
        # Modify context2
        context2.add_user_message("Context 2 message")
        context2.set_metadata("context", "2")
        
        # Verify isolation
        self.assertEqual(len(context1.get_conversation_history()), 1)
        self.assertEqual(len(context2.get_conversation_history()), 1)
        self.assertEqual(context1.get_metadata("context"), "1")
        self.assertEqual(context2.get_metadata("context"), "2")

    def test_task_context_performance_metrics(self):
        """Test TaskContext tracks performance metrics."""
        from common.task_context import TaskContext
        
        context = TaskContext(task_graph=self.task_graph)
        
        # Get initial metrics
        metrics = context.get_performance_metrics()
        self.assertIn('total_tasks', metrics)
        self.assertIn('completed_tasks', metrics)
        self.assertIn('failed_tasks', metrics)
        self.assertIn('execution_time', metrics)
        
        # Complete a task and check metrics update
        task_id = context.task_graph.task_name_map["planning_task"]
        context.complete_task(task_id, "Task completed")
        
        updated_metrics = context.get_performance_metrics()
        self.assertEqual(updated_metrics['completed_tasks'], 1)


if __name__ == '__main__':
    unittest.main()