"""Unit tests for Intent Processor WorkflowExecutionIntent handling.

Tests the Intent Processor's ability to handle WorkflowExecutionIntent processing
following Document 35 Phase 2 implementation for LLM-safe architecture compliance.

Following Documents 25 & 26 LLM-safe architecture patterns.
"""

import time
from unittest.mock import Mock, patch

import pytest

from common.intent_processor import IntentProcessor
from common.intents import NotificationIntent
from common.workflow_intent import WorkflowExecutionIntent


class TestIntentProcessorWorkflowHandling:
    """Test Intent Processor WorkflowExecutionIntent handling."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_communication_manager = Mock()
        self.mock_workflow_engine = Mock()
        self.mock_message_bus = Mock()

        # Create intent processor
        self.intent_processor = IntentProcessor(
            communication_manager=self.mock_communication_manager,
            workflow_engine=self.mock_workflow_engine,
            message_bus=self.mock_message_bus,
        )

        # Create sample WorkflowExecutionIntent
        self.sample_intent = WorkflowExecutionIntent(
            tasks=[
                {
                    "id": "search_task",
                    "name": "Search Information",
                    "description": "Search for information",
                    "role": "search",
                    "parameters": {"query": "test query"},
                },
                {
                    "id": "weather_task",
                    "name": "Get Weather",
                    "description": "Check weather conditions",
                    "role": "weather",
                    "parameters": {"location": "Chicago"},
                },
            ],
            dependencies=[
                {
                    "source_task_id": "search_task",
                    "target_task_id": "weather_task",
                    "type": "sequential",
                }
            ],
            request_id="test_request_123",
            user_id="test_user",
            channel_id="slack:C123",
            original_instruction="Test workflow execution",
        )

    def test_workflow_intent_processing_delegates_to_workflow_engine(self):
        """Test that WorkflowExecutionIntent processing delegates to workflow engine."""
        # Act
        self.intent_processor._process_workflow(self.sample_intent)

        # Assert
        self.mock_workflow_engine.execute_workflow_intent.assert_called_once_with(
            self.sample_intent
        )

    def test_workflow_intent_processing_without_workflow_engine(self):
        """Test graceful handling when no workflow engine available."""
        # Arrange
        self.intent_processor.workflow_engine = None

        # Act - Should not raise exception
        self.intent_processor._process_workflow(self.sample_intent)

        # Assert - No workflow engine call should be made
        self.mock_workflow_engine.execute_workflow_intent.assert_not_called()

    def test_workflow_intent_processing_with_invalid_intent(self):
        """Test handling of invalid WorkflowExecutionIntent."""
        # Arrange
        invalid_intent = WorkflowExecutionIntent(
            tasks=[],  # Empty tasks
            dependencies=[],
            request_id="",  # Empty request_id
            user_id="test_user",
            channel_id="console",
            original_instruction="Invalid intent test",
        )

        # Act - Should still process (validation happens elsewhere)
        self.intent_processor._process_workflow(invalid_intent)

        # Assert
        self.mock_workflow_engine.execute_workflow_intent.assert_called_once_with(
            invalid_intent
        )

    def test_workflow_intent_processing_preserves_intent_data(self):
        """Test that intent data is preserved during processing."""
        # Act
        self.intent_processor._process_workflow(self.sample_intent)

        # Assert
        call_args = self.mock_workflow_engine.execute_workflow_intent.call_args[0][0]
        assert call_args.request_id == self.sample_intent.request_id
        assert call_args.user_id == self.sample_intent.user_id
        assert call_args.channel_id == self.sample_intent.channel_id
        assert len(call_args.tasks) == len(self.sample_intent.tasks)
        assert len(call_args.dependencies) == len(self.sample_intent.dependencies)

    def test_workflow_intent_processing_error_handling(self):
        """Test error handling in workflow intent processing."""
        # Arrange
        self.mock_workflow_engine.execute_workflow_intent.side_effect = Exception(
            "Test error"
        )

        # Act - Should not raise exception (error handling in intent processor)
        self.intent_processor._process_workflow(self.sample_intent)

        # Assert
        self.mock_workflow_engine.execute_workflow_intent.assert_called_once()

    def test_workflow_intent_registration_in_core_handlers(self):
        """Test that WorkflowExecutionIntent is registered in core handlers."""
        # Assert
        assert WorkflowExecutionIntent in self.intent_processor._core_handlers
        assert (
            self.intent_processor._core_handlers[WorkflowExecutionIntent]
            == self.intent_processor._process_workflow
        )

    def test_process_intents_handles_workflow_execution_intent(self):
        """Test that process_intents method handles WorkflowExecutionIntent."""
        # Arrange
        intents = [self.sample_intent]

        # Act
        result = self.intent_processor.process_intents(intents)

        # Assert
        assert result["processed"] == 1
        assert result["failed"] == 0
        assert len(result["errors"]) == 0
        self.mock_workflow_engine.execute_workflow_intent.assert_called_once()

    def test_process_intents_handles_mixed_intent_types(self):
        """Test processing mixed intent types including WorkflowExecutionIntent."""
        # Arrange
        notification_intent = NotificationIntent(
            message="Test notification", channel="console", priority="medium"
        )

        intents = [self.sample_intent, notification_intent]

        # Act
        result = self.intent_processor.process_intents(intents)

        # Assert
        assert result["processed"] == 2
        assert result["failed"] == 0
        self.mock_workflow_engine.execute_workflow_intent.assert_called_once()

    def test_workflow_intent_validation_before_processing(self):
        """Test that intent validation occurs before processing."""
        # Arrange
        invalid_intent = WorkflowExecutionIntent(
            tasks=[],  # Empty tasks should fail validation
            dependencies=[],
            request_id="",  # Empty request_id should fail validation
            user_id="test_user",
            channel_id="console",
            original_instruction="Invalid intent",
        )

        intents = [invalid_intent]

        # Act
        result = self.intent_processor.process_intents(intents)

        # Assert - Invalid intent should be marked as failed
        assert result["processed"] == 0
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        self.mock_workflow_engine.execute_workflow_intent.assert_not_called()

    def test_workflow_intent_processing_is_synchronous(self):
        """Test that workflow intent processing is synchronous (LLM-safe)."""
        # Arrange
        start_time = time.time()

        # Act
        self.intent_processor._process_workflow(self.sample_intent)

        # Assert
        execution_time = time.time() - start_time
        assert execution_time < 0.1  # Should be very fast (synchronous)
        self.mock_workflow_engine.execute_workflow_intent.assert_called_once()

    def test_multiple_workflow_intents_processing(self):
        """Test processing multiple WorkflowExecutionIntent objects."""
        # Arrange
        intent1 = WorkflowExecutionIntent(
            tasks=[
                {
                    "id": "task_1",
                    "name": "Task 1",
                    "description": "First task",
                    "role": "search",
                }
            ],
            dependencies=[],
            request_id="request_1",
            user_id="user_1",
            channel_id="console",
            original_instruction="First workflow",
        )

        intent2 = WorkflowExecutionIntent(
            tasks=[
                {
                    "id": "task_2",
                    "name": "Task 2",
                    "description": "Second task",
                    "role": "weather",
                }
            ],
            dependencies=[],
            request_id="request_2",
            user_id="user_2",
            channel_id="slack:C456",
            original_instruction="Second workflow",
        )

        intents = [intent1, intent2]

        # Act
        result = self.intent_processor.process_intents(intents)

        # Assert
        assert result["processed"] == 2
        assert result["failed"] == 0
        assert self.mock_workflow_engine.execute_workflow_intent.call_count == 2

        # Verify both intents were processed
        call_args_list = (
            self.mock_workflow_engine.execute_workflow_intent.call_args_list
        )
        processed_request_ids = [call[0][0].request_id for call in call_args_list]
        assert "request_1" in processed_request_ids
        assert "request_2" in processed_request_ids
