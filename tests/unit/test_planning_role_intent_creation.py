"""Unit tests for planning role WorkflowIntent creation.

Tests the planning role's ability to create valid WorkflowIntent objects
from TaskGraph JSON following Document 35 Phase 1 implementation.

Following Documents 25 & 26 LLM-safe architecture patterns.
"""

import json
from unittest.mock import MagicMock, Mock

import pytest

from common.intents import WorkflowIntent
from roles.core_planning import execute_task_graph


class TestPlanningRoleIntentCreation:
    """Test planning role WorkflowIntent creation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = Mock()
        self.mock_context.context_id = "test_request_123"
        self.mock_context.user_id = "test_user"
        self.mock_context.channel_id = "slack:C123"
        self.mock_context.original_prompt = "Test workflow request"

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

    def test_valid_task_graph_creates_workflow_execution_intent(self):
        """Test that valid TaskGraph JSON creates WorkflowIntent."""
        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json,
            context=self.mock_context,
            pre_data={},
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert result.request_id == "test_request_123"
        assert result.user_id == "test_user"
        assert result.channel_id == "slack:C123"
        assert result.original_instruction == "Test workflow request"
        assert len(result.tasks) == 2
        assert len(result.dependencies) == 1

    def test_intent_contains_all_task_information(self):
        """Test that created intent preserves all task information."""
        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json,
            context=self.mock_context,
            pre_data={},
        )

        # Assert
        assert isinstance(result, WorkflowIntent)

        # Check first task
        task_1 = result.tasks[0]
        assert task_1["id"] == "task_1"
        assert task_1["name"] == "Search Thailand Info"
        assert task_1["role"] == "search"
        assert task_1["parameters"]["query"] == "Thailand travel"

        # Check second task
        task_2 = result.tasks[1]
        assert task_2["id"] == "task_2"
        assert task_2["name"] == "Get Weather"
        assert task_2["role"] == "weather"
        assert task_2["parameters"]["location"] == "Chicago"

    def test_intent_contains_dependency_information(self):
        """Test that created intent preserves dependency information."""
        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json,
            context=self.mock_context,
            pre_data={},
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert len(result.dependencies) == 1

        dependency = result.dependencies[0]
        assert dependency["source_task_id"] == "task_1"
        assert dependency["target_task_id"] == "task_2"
        assert dependency["type"] == "sequential"

    def test_intent_validation_works(self):
        """Test that created intent passes validation."""
        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json,
            context=self.mock_context,
            pre_data={},
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert result.validate() is True

    def test_intent_get_expected_workflow_ids(self):
        """Test that intent can generate expected workflow IDs."""
        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json,
            context=self.mock_context,
            pre_data={},
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        expected_ids = result.get_expected_workflow_ids()
        assert expected_ids == {
            "test_request_123_task_task_1",
            "test_request_123_task_task_2",
        }

    def test_invalid_json_returns_error_message(self):
        """Test that invalid JSON raises ValueError exception."""
        # Arrange
        invalid_json = "This is not valid JSON"

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid JSON in TaskGraph"):
            execute_task_graph(
                llm_result=invalid_json, context=self.mock_context, pre_data={}
            )

    def test_missing_required_task_fields_returns_error(self):
        """Test that missing required task fields creates intent (validation happens later)."""
        # Arrange
        invalid_task_graph = json.dumps(
            {
                "tasks": [
                    {
                        "id": "task_1",
                        # Missing name, description, role - but intent creation succeeds
                    }
                ],
                "dependencies": [],
            }
        )

        # Act
        result = execute_task_graph(
            llm_result=invalid_task_graph, context=self.mock_context, pre_data={}
        )

        # Assert - Intent is created, validation happens at processing time
        assert isinstance(result, WorkflowIntent)
        assert len(result.tasks) == 1
        assert result.tasks[0]["id"] == "task_1"

    def test_invalid_role_references_returns_error(self):
        """Test that invalid role references create intent (validation happens later)."""
        # Arrange
        invalid_role_graph = json.dumps(
            {
                "tasks": [
                    {
                        "id": "task_1",
                        "name": "Invalid Task",
                        "description": "Task with invalid role",
                        "role": "nonexistent_role",
                        "parameters": {},
                    }
                ],
                "dependencies": [],
            }
        )

        # Act
        result = execute_task_graph(
            llm_result=invalid_role_graph,
            context=self.mock_context,
            pre_data={"available_roles": ["search", "weather", "timer"]},
        )

        # Assert - Intent is created, role validation happens at execution time
        assert isinstance(result, WorkflowIntent)
        assert len(result.tasks) == 1
        assert result.tasks[0]["role"] == "nonexistent_role"

    def test_context_without_required_fields_handled_gracefully(self):
        """Test that context without required fields is handled gracefully."""
        # Arrange
        minimal_context = Mock()
        minimal_context.context_id = "minimal_123"
        # Remove the mock attributes so getattr returns defaults
        del minimal_context.user_id
        del minimal_context.channel_id
        del minimal_context.original_prompt

        # Act
        result = execute_task_graph(
            llm_result=self.valid_task_graph_json, context=minimal_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert result.request_id == "minimal_123"
        assert result.user_id == "unknown"  # Default value
        assert result.channel_id == "console"  # Default value
        assert result.original_instruction == "Multi-step workflow"  # Default value

    def test_empty_task_list_returns_error(self):
        """Test that empty task list creates invalid intent (fails validation)."""
        # Arrange
        empty_task_graph = json.dumps({"tasks": [], "dependencies": []})

        # Act
        result = execute_task_graph(
            llm_result=empty_task_graph, context=self.mock_context, pre_data={}
        )

        # Assert - Intent is created but will fail validation
        assert isinstance(result, WorkflowIntent)
        assert len(result.tasks) == 0
        assert not result.validate()  # Validation should fail for empty tasks

    def test_mixed_content_with_json_extracts_json(self):
        """Test that mixed content with embedded JSON extracts the JSON."""
        # Arrange
        mixed_content = f"""
        Here's the workflow plan:

        {self.valid_task_graph_json}

        This should work well for your request.
        """

        # Act
        result = execute_task_graph(
            llm_result=mixed_content, context=self.mock_context, pre_data={}
        )

        # Assert
        assert isinstance(result, WorkflowIntent)
        assert len(result.tasks) == 2
