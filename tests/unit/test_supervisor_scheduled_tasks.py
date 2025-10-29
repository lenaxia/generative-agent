"""Unit tests for Supervisor scheduled task management.

Tests the Supervisor's ability to manage scheduled tasks following Document 35 Phase 2
implementation for LLM-safe architecture compliance.

Following Documents 25 & 26 LLM-safe architecture patterns.
"""

import time
from unittest.mock import Mock, patch

import pytest

from supervisor.supervisor import Supervisor


class TestSupervisorScheduledTasks:
    """Test Supervisor scheduled task management."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create supervisor with main config file
        self.supervisor = Supervisor("config.yaml")

        # Mock task handlers
        self.mock_handler = Mock()
        self.mock_interval_handler = Mock()

    def test_add_scheduled_task_one_time(self):
        """Test adding one-time scheduled task."""
        # Arrange
        task = {
            "type": "test_task",
            "handler": self.mock_handler,
            "data": {"key": "value"},
        }

        # Act
        self.supervisor.add_scheduled_task(task)

        # Assert
        assert len(self.supervisor._scheduled_tasks) == 1
        assert self.supervisor._scheduled_tasks[0] == task

    def test_add_scheduled_task_interval(self):
        """Test adding interval scheduled task."""
        # Arrange
        task = {
            "type": "interval_task",
            "handler": self.mock_interval_handler,
            "interval": 60,  # 60 seconds
            "data": {"interval_key": "interval_value"},
        }

        # Act
        self.supervisor.add_scheduled_task(task)

        # Assert
        assert len(self.supervisor._scheduled_tasks) == 1
        assert self.supervisor._scheduled_tasks[0] == task

    def test_process_scheduled_tasks_one_time(self):
        """Test processing one-time scheduled tasks."""
        # Arrange
        task = {
            "type": "test_task",
            "handler": self.mock_handler,
            "data": {"test": "data"},
        }
        self.supervisor.add_scheduled_task(task)

        # Act
        self.supervisor.process_scheduled_tasks()

        # Assert
        self.mock_handler.assert_called_once_with({"test": "data"})
        assert len(self.supervisor._scheduled_tasks) == 0  # One-time task removed

    def test_process_scheduled_tasks_with_intent(self):
        """Test processing scheduled tasks with intent parameter."""
        # Arrange
        from common.intents import WorkflowIntent

        intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[
                {
                    "id": "task_1",
                    "name": "Test",
                    "description": "Test",
                    "role": "search",
                }
            ],
            dependencies=[],
            request_id="test_123",
            user_id="test_user",
            channel_id="console",
            original_instruction="Test intent",
        )

        task = {
            "type": "process_workflow_intent",
            "handler": self.mock_handler,
            "intent": intent,
        }
        self.supervisor.add_scheduled_task(task)

        # Act
        self.supervisor.process_scheduled_tasks()

        # Assert
        self.mock_handler.assert_called_once_with(intent)
        assert len(self.supervisor._scheduled_tasks) == 0

    def test_process_scheduled_tasks_interval(self):
        """Test processing interval scheduled tasks."""
        # Arrange
        task = {
            "type": "interval_task",
            "handler": self.mock_interval_handler,
            "interval": 1,  # 1 second for testing
        }
        self.supervisor.add_scheduled_task(task)

        # Act - First call should execute
        self.supervisor.process_scheduled_tasks()

        # Assert
        self.mock_interval_handler.assert_called_once()
        assert len(self.supervisor._scheduled_tasks) == 1  # Interval task remains
        assert "interval_task" in self.supervisor._scheduled_intervals

        # Act - Second call immediately should not execute (interval not met)
        self.mock_interval_handler.reset_mock()
        self.supervisor.process_scheduled_tasks()

        # Assert
        self.mock_interval_handler.assert_not_called()

    def test_process_scheduled_tasks_interval_after_time(self):
        """Test processing interval tasks after interval has passed."""
        # Arrange
        task = {
            "type": "interval_task",
            "handler": self.mock_interval_handler,
            "interval": 0.1,  # 0.1 seconds for testing
        }
        self.supervisor.add_scheduled_task(task)

        # Act - First execution
        self.supervisor.process_scheduled_tasks()
        self.mock_interval_handler.assert_called_once()

        # Wait for interval to pass
        time.sleep(0.2)

        # Act - Second execution after interval
        self.mock_interval_handler.reset_mock()
        self.supervisor.process_scheduled_tasks()

        # Assert
        self.mock_interval_handler.assert_called_once()

    def test_process_scheduled_tasks_error_handling(self):
        """Test error handling in scheduled task processing."""

        # Arrange
        def failing_handler(task):
            raise Exception("Test error")

        task = {
            "type": "failing_task",
            "handler": failing_handler,
            "data": {"test": "data"},
        }
        self.supervisor.add_scheduled_task(task)

        # Act - Should not raise exception
        self.supervisor.process_scheduled_tasks()

        # Assert - Task should be removed even if it failed
        assert len(self.supervisor._scheduled_tasks) == 0

    def test_process_scheduled_tasks_multiple_tasks(self):
        """Test processing multiple scheduled tasks."""
        # Arrange
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()

        tasks = [
            {"type": "task_1", "handler": handler1},
            {"type": "task_2", "handler": handler2},
            {"type": "task_3", "handler": handler3},
        ]

        for task in tasks:
            self.supervisor.add_scheduled_task(task)

        # Act
        self.supervisor.process_scheduled_tasks()

        # Assert
        handler1.assert_called_once()
        handler2.assert_called_once()
        handler3.assert_called_once()
        assert len(self.supervisor._scheduled_tasks) == 0

    def test_scheduled_task_types(self):
        """Test different scheduled task types."""
        # Arrange
        one_time_task = {"type": "one_time", "handler": Mock()}
        interval_task = {"type": "interval", "handler": Mock(), "interval": 60}
        intent_task = {"type": "process_intent", "handler": Mock(), "intent": Mock()}

        # Act
        self.supervisor.add_scheduled_task(one_time_task)
        self.supervisor.add_scheduled_task(interval_task)
        self.supervisor.add_scheduled_task(intent_task)

        # Assert
        assert len(self.supervisor._scheduled_tasks) == 3

        # Verify task types
        task_types = [task["type"] for task in self.supervisor._scheduled_tasks]
        assert "one_time" in task_types
        assert "interval" in task_types
        assert "process_intent" in task_types

    def test_scheduled_task_data_preservation(self):
        """Test that scheduled task data is preserved correctly."""
        # Arrange
        test_data = {
            "complex_data": {
                "nested": {"value": 123},
                "array": [1, 2, 3],
                "string": "test",
            }
        }

        task = {"type": "data_test", "handler": self.mock_handler, "data": test_data}

        # Act
        self.supervisor.add_scheduled_task(task)
        self.supervisor.process_scheduled_tasks()

        # Assert
        self.mock_handler.assert_called_once_with(test_data)
        call_args = self.mock_handler.call_args[0][0]
        assert call_args == test_data
