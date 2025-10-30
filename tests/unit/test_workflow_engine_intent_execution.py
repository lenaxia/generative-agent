"""Unit tests for Workflow Engine WorkflowIntent execution.

Tests the workflow engine's ability to execute WorkflowIntent with task graphs
following Document 35 Phase 2.3 implementation.

Following Documents 25 & 26 LLM-safe architecture patterns.
"""

from unittest.mock import Mock

from common.intents import WorkflowIntent
from common.message_bus import MessageBus
from supervisor.workflow_engine import WorkflowEngine


class TestWorkflowEngineIntentExecution:
    """Test workflow engine WorkflowIntent execution."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_llm_factory = Mock()
        self.mock_message_bus = MessageBus()
        self.mock_message_bus.start()

        # Create workflow engine
        self.workflow_engine = WorkflowEngine(
            llm_factory=self.mock_llm_factory,
            message_bus=self.mock_message_bus,
        )

        # Create sample WorkflowIntent with task graph
        self.sample_intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[
                {
                    "id": "task_1",
                    "name": "Search Task",
                    "description": "Search for information",
                    "role": "search",
                    "parameters": {"query": "test"},
                }
            ],
            dependencies=[],
            request_id="test_request_123",
            user_id="test_user",
            channel_id="console",
            original_instruction="Test workflow",
        )

    def test_execute_workflow_intent_creates_task_graph(self):
        """Test that execute_workflow_intent creates TaskGraph from intent."""
        # Arrange
        self.workflow_engine.universal_agent.execute_task = Mock(
            return_value="Task completed"
        )

        # Act
        result = self.workflow_engine.execute_workflow_intent(self.sample_intent)

        # Assert
        assert result == self.sample_intent.request_id
        assert self.workflow_engine.universal_agent.execute_task.called

    def test_execute_workflow_intent_executes_successfully(self):
        """Test that workflow intent executes without errors."""
        # Arrange
        self.workflow_engine.universal_agent.execute_task = Mock(
            return_value="Task completed"
        )

        # Act
        result = self.workflow_engine.execute_workflow_intent(self.sample_intent)

        # Assert
        assert result == self.sample_intent.request_id

    def test_execute_workflow_intent_with_multiple_tasks(self):
        """Test executing intent with multiple tasks."""
        # Arrange
        multi_task_intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[
                {
                    "id": "task_1",
                    "name": "Task 1",
                    "description": "First task",
                    "role": "search",
                },
                {
                    "id": "task_2",
                    "name": "Task 2",
                    "description": "Second task",
                    "role": "weather",
                },
            ],
            dependencies=[
                {
                    "source_task_id": "task_1",
                    "target_task_id": "task_2",
                    "type": "sequential",
                }
            ],
            request_id="multi_task_123",
            user_id="test_user",
            channel_id="console",
            original_instruction="Multi-task workflow",
        )

        self.workflow_engine.universal_agent.execute_task = Mock(
            return_value="Task completed"
        )

        # Act
        result = self.workflow_engine.execute_workflow_intent(multi_task_intent)

        # Assert
        assert result == multi_task_intent.request_id

    def test_execute_workflow_intent_handles_errors_gracefully(self):
        """Test error handling in workflow intent execution."""
        # Arrange
        self.workflow_engine.universal_agent.execute_task = Mock(
            side_effect=Exception("Test error")
        )

        # Act - Should not raise exception
        result = self.workflow_engine.execute_workflow_intent(self.sample_intent)

        # Assert - Should return request_id even on error
        assert result == self.sample_intent.request_id

    def test_execute_workflow_intent_validates_intent(self):
        """Test that invalid intents are handled."""
        # Arrange
        invalid_intent = WorkflowIntent(
            workflow_type="task_graph_execution",
            parameters={},
            tasks=[],  # Empty tasks
            dependencies=[],
            request_id="",  # Empty request_id
            user_id="test_user",
            channel_id="console",
            original_instruction="Invalid",
        )

        # Act
        result = self.workflow_engine.execute_workflow_intent(invalid_intent)

        # Assert - Should handle gracefully
        assert result is not None
