"""Integration tests for Document 35 intent-based planning implementation.

Tests the integration between planning role intent creation and universal agent
intent detection following Document 35 Phase 1 implementation.

Following Documents 25 & 26 LLM-safe architecture patterns.
"""

import json
from unittest.mock import Mock, patch

import pytest

from common.workflow_intent import WorkflowExecutionIntent
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent
from roles.core_planning import execute_task_graph


class TestIntentBasedPlanningIntegration:
    """Integration tests for intent-based planning workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.valid_task_graph_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "task_1",
                        "name": "Search Thailand Info",
                        "description": "Search for Thailand travel information",
                        "role": "search",
                        "parameters": {"query": "Thailand travel"},
                    },
                    {
                        "id": "task_2",
                        "name": "Get Weather",
                        "description": "Check weather in Chicago",
                        "role": "weather",
                        "parameters": {"location": "Chicago"},
                    },
                ],
                "dependencies": [
                    {
                        "source_task_id": "task_1",
                        "target_task_id": "task_2",
                        "type": "sequential",
                    }
                ],
            }
        )

    def test_planning_role_creates_valid_intent(self):
        """Test that planning role creates valid WorkflowExecutionIntent."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "integration_test_123"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "slack:C123"
        mock_context.original_prompt = "Plan Thailand trip and check weather"

        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json, context=mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowExecutionIntent)
        assert result.validate() is True
        assert len(result.tasks) == 2
        assert len(result.dependencies) == 1
        assert result.request_id == "integration_test_123"

    def test_intent_contains_expected_workflow_ids(self):
        """Test that intent generates correct workflow IDs for lifecycle tracking."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "lifecycle_test_456"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Multi-step workflow"

        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json, context=mock_context, pre_data={}
        )

        # Assert
        expected_workflow_ids = result.get_expected_workflow_ids()
        assert expected_workflow_ids == {
            "lifecycle_test_456_task_task_1",
            "lifecycle_test_456_task_task_2",
        }

    def test_intent_serialization_and_deserialization(self):
        """Test that WorkflowExecutionIntent can be serialized and deserialized."""
        # Arrange
        mock_context = Mock()
        mock_context.context_id = "serialization_test_789"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "slack:C456"
        mock_context.original_prompt = "Serialization test workflow"

        # Act
        original_intent = execute_task_graph(
            llm_result=self.valid_task_graph_json, context=mock_context, pre_data={}
        )

        # Serialize and deserialize
        serialized = original_intent.to_dict()
        deserialized_intent = WorkflowExecutionIntent.from_dict(serialized)

        # Assert
        assert deserialized_intent.request_id == original_intent.request_id
        assert deserialized_intent.tasks == original_intent.tasks
        assert deserialized_intent.dependencies == original_intent.dependencies
        assert deserialized_intent.user_id == original_intent.user_id
        assert deserialized_intent.channel_id == original_intent.channel_id

    def test_mixed_content_json_extraction(self):
        """Test that mixed content with embedded JSON is handled correctly."""
        # Arrange
        mixed_content = f"""
        Here's the workflow plan for your request:

        {self.valid_task_graph_json}

        This should handle your Thailand trip planning and weather check.
        """

        mock_context = Mock()
        mock_context.context_id = "mixed_content_test_101"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Mixed content test"

        # Act
        result = execute_task_graph(
            llm_result=mixed_content, context=mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowExecutionIntent)
        assert len(result.tasks) == 2
        assert result.tasks[0]["name"] == "Search Thailand Info"
        assert result.tasks[1]["name"] == "Get Weather"

    def test_intent_validation_catches_invalid_intents(self):
        """Test that intent validation works correctly."""
        # Arrange
        invalid_task_graph = json.dumps(
            {"tasks": [], "dependencies": []}  # Empty tasks
        )

        mock_context = Mock()
        mock_context.context_id = ""  # Empty request_id
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Invalid intent test"

        # Act
        result = execute_task_graph(
            llm_result=invalid_task_graph, context=mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowExecutionIntent)
        assert (
            result.validate() is False
        )  # Should fail validation due to empty tasks and request_id

    def test_error_handling_for_malformed_json(self):
        """Test error handling for malformed JSON input."""
        # Arrange
        malformed_json = (
            '{"tasks": [{"id": "task_1", "name": "Test"'  # Missing closing braces
        )

        mock_context = Mock()
        mock_context.context_id = "error_test_202"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Error handling test"

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid JSON in TaskGraph"):
            execute_task_graph(
                llm_result=malformed_json, context=mock_context, pre_data={}
            )

    def test_error_handling_for_validation_error_messages(self):
        """Test error handling for validation error messages."""
        # Arrange
        error_message = (
            "Invalid role references: Task 'task_1' uses unavailable role 'nonexistent'"
        )

        mock_context = Mock()
        mock_context.context_id = "validation_error_test_303"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "console"
        mock_context.original_prompt = "Validation error test"

        # Act & Assert
        with pytest.raises(
            ValueError, match="Cannot create WorkflowExecutionIntent from error message"
        ):
            execute_task_graph(
                llm_result=error_message, context=mock_context, pre_data={}
            )

    def test_intent_preserves_task_parameters(self):
        """Test that intent preserves task parameters for role execution."""
        # Arrange
        task_graph_with_params = json.dumps(
            {
                "tasks": [
                    {
                        "id": "weather_task",
                        "name": "Get Current Weather",
                        "description": "Check current weather conditions",
                        "role": "weather",
                        "parameters": {
                            "location": "Chicago",
                            "timeframe": "current",
                            "format": "detailed",
                        },
                    }
                ],
                "dependencies": [],
            }
        )

        mock_context = Mock()
        mock_context.context_id = "params_test_404"
        mock_context.user_id = "test_user"
        mock_context.channel_id = "slack:C789"
        mock_context.original_prompt = "Parameter preservation test"

        # Act
        result = execute_task_graph(
            llm_result=task_graph_with_params, context=mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowExecutionIntent)
        weather_task = result.tasks[0]
        assert weather_task["parameters"]["location"] == "Chicago"
        assert weather_task["parameters"]["timeframe"] == "current"
        assert weather_task["parameters"]["format"] == "detailed"
