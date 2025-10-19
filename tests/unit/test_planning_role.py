"""Unit tests for the enhanced planning role implementation.

Tests for Document 34: Planning Role Design & Implementation
Following TDD approach with comprehensive coverage of:
- Role loading and filtering
- TaskGraph validation
- JSON parsing and structure validation
- Role reference validation
- End-to-end planning flow
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import functions that will be implemented
from roles.core_planning import (
    ROLE_CONFIG,
    _extract_available_role_names,
    _format_roles_for_prompt,
    _validate_dependency_structure,
    _validate_role_references,
    _validate_task_graph_structure,
    _validate_task_structure,
    load_available_roles,
    register_role,
    validate_task_graph,
)


class TestRoleConfiguration:
    """Test the enhanced role configuration."""

    def test_role_config_structure(self):
        """Test that ROLE_CONFIG has required fields."""
        assert ROLE_CONFIG["name"] == "planning"
        assert ROLE_CONFIG["version"] == "3.0.0"
        assert ROLE_CONFIG["llm_type"] == "STRONG"
        assert ROLE_CONFIG["fast_reply"] is False
        assert "when_to_use" in ROLE_CONFIG
        assert "tools" in ROLE_CONFIG
        assert "prompts" in ROLE_CONFIG

    def test_role_config_tools_disabled(self):
        """Test that tools are properly disabled."""
        tools_config = ROLE_CONFIG["tools"]
        assert tools_config["automatic"] is False
        assert tools_config["shared"] == []
        assert tools_config["include_builtin"] is False

    def test_role_config_prompt_structure(self):
        """Test that system prompt contains required elements."""
        system_prompt = ROLE_CONFIG["prompts"]["system"]
        assert "{{available_roles}}" in system_prompt
        assert "TaskGraph" in system_prompt
        assert "BNF GRAMMAR" in system_prompt
        assert "EXAMPLE OUTPUT" in system_prompt


class TestRoleLoading:
    """Test role discovery and loading functionality."""

    @patch("llm_provider.role_registry.RoleRegistry.get_global_registry")
    def test_load_available_roles_success(self, mock_registry):
        """Test successful loading and filtering of available roles."""
        # Mock role registry with various roles
        mock_roles = {
            "weather": Mock(
                config={
                    "description": "Weather information and forecasts",
                    "when_to_use": "Get weather data",
                    "parameters": {"location": {"type": "string"}},
                }
            ),
            "timer": Mock(
                config={
                    "description": "Timer management",
                    "when_to_use": "Set timers and alarms",
                    "parameters": {"duration": {"type": "string"}},
                }
            ),
            "planning": Mock(
                config={"description": "Task planning"}
            ),  # Should be filtered out
            "router": Mock(
                config={"description": "Request routing"}
            ),  # Should be filtered out
        }
        mock_registry.return_value.roles = mock_roles

        result = load_available_roles("Plan my day", Mock(), {})

        assert "available_roles" in result
        available_roles_text = result["available_roles"]
        assert "weather" in available_roles_text
        assert "timer" in available_roles_text
        assert "planning" not in available_roles_text
        assert "router" not in available_roles_text

    @patch("llm_provider.role_registry.RoleRegistry.get_global_registry")
    def test_load_available_roles_error_handling(self, mock_registry):
        """Test error handling when role loading fails."""
        mock_registry.side_effect = Exception("Registry error")

        result = load_available_roles("test", Mock(), {})

        assert "available_roles" in result
        assert "Error loading roles" in result["available_roles"]
        assert "weather, timer, conversation" in result["available_roles"]

    def test_format_roles_for_prompt(self):
        """Test role formatting for prompt injection."""
        role_info = [
            {
                "name": "weather",
                "description": "Weather information",
                "when_to_use": "Get weather data",
                "parameters": {"location": {"type": "string"}},
            },
            {
                "name": "timer",
                "description": "Timer management",
                "when_to_use": "",
                "parameters": {},
            },
        ]

        result = _format_roles_for_prompt(role_info)

        assert "**weather**:" in result
        assert "Weather information" in result
        assert "When to use: Get weather data" in result
        assert "Parameters:" in result
        assert "**timer**:" in result
        assert "Timer management" in result


class TestTaskGraphValidation:
    """Test TaskGraph structure validation."""

    def test_valid_task_graph_structure(self):
        """Test validation of a valid TaskGraph."""
        valid_graph = {
            "tasks": [
                {
                    "id": "task_1",
                    "name": "Check Weather",
                    "description": "Get current weather",
                    "role": "weather",
                    "parameters": {},
                }
            ],
            "dependencies": [],
        }

        errors = _validate_task_graph_structure(valid_graph)
        assert len(errors) == 0

    def test_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        # Missing tasks
        invalid_graph1 = {"dependencies": []}
        errors1 = _validate_task_graph_structure(invalid_graph1)
        assert "Missing 'tasks' array" in errors1

        # Missing dependencies
        invalid_graph2 = {"tasks": []}
        errors2 = _validate_task_graph_structure(invalid_graph2)
        assert "Missing 'dependencies' array" in errors2

    def test_empty_tasks_validation(self):
        """Test validation fails for empty tasks array."""
        invalid_graph = {"tasks": [], "dependencies": []}
        errors = _validate_task_graph_structure(invalid_graph)
        assert "At least one task is required" in errors

    def test_invalid_tasks_type(self):
        """Test validation fails for non-array tasks."""
        invalid_graph = {"tasks": "not_an_array", "dependencies": []}
        errors = _validate_task_graph_structure(invalid_graph)
        assert "'tasks' must be an array" in errors

    def test_invalid_dependencies_type(self):
        """Test validation fails for non-array dependencies."""
        invalid_graph = {
            "tasks": [
                {"id": "task_1", "name": "Test", "description": "Test", "role": "test"}
            ],
            "dependencies": "not_an_array",
        }
        errors = _validate_task_graph_structure(invalid_graph)
        assert "'dependencies' must be an array" in errors


class TestTaskValidation:
    """Test individual task validation."""

    def test_valid_task_structure(self):
        """Test validation of valid task structure."""
        valid_task = {
            "id": "task_1",
            "name": "Test Task",
            "description": "A test task",
            "role": "weather",
        }

        errors = _validate_task_structure(valid_task, 0)
        assert len(errors) == 0

    def test_missing_required_task_fields(self):
        """Test validation fails for missing required task fields."""
        invalid_task = {
            "id": "task_1",
            "name": "Test Task"
            # Missing description and role
        }

        errors = _validate_task_structure(invalid_task, 0)
        assert "Task 0: missing 'description'" in errors
        assert "Task 0: missing 'role'" in errors

    def test_empty_task_fields(self):
        """Test validation fails for empty task fields."""
        invalid_task = {
            "id": "",
            "name": "   ",
            "description": "Valid description",
            "role": "weather",
        }

        errors = _validate_task_structure(invalid_task, 1)
        assert "Task 1: 'id' must be non-empty string" in errors
        assert "Task 1: 'name' must be non-empty string" in errors


class TestDependencyValidation:
    """Test dependency validation."""

    def test_valid_dependency_structure(self):
        """Test validation of valid dependency structure."""
        task_ids = {"task_1", "task_2"}
        valid_dependency = {
            "source_task_id": "task_1",
            "target_task_id": "task_2",
            "type": "sequential",
        }

        errors = _validate_dependency_structure(valid_dependency, 0, task_ids)
        assert len(errors) == 0

    def test_missing_dependency_fields(self):
        """Test validation fails for missing dependency fields."""
        task_ids = {"task_1", "task_2"}
        invalid_dependency = {
            "source_task_id": "task_1"
            # Missing target_task_id and type
        }

        errors = _validate_dependency_structure(invalid_dependency, 0, task_ids)
        assert "Dependency 0: missing 'target_task_id'" in errors
        assert "Dependency 0: missing 'type'" in errors

    def test_invalid_task_references(self):
        """Test validation fails for invalid task references."""
        task_ids = {"task_1", "task_2"}
        invalid_dependency = {
            "source_task_id": "nonexistent_task",
            "target_task_id": "another_nonexistent",
            "type": "sequential",
        }

        errors = _validate_dependency_structure(invalid_dependency, 0, task_ids)
        assert "Dependency 0: invalid 'source_task_id'" in errors
        assert "Dependency 0: invalid 'target_task_id'" in errors

    def test_invalid_dependency_type(self):
        """Test validation fails for invalid dependency type."""
        task_ids = {"task_1", "task_2"}
        invalid_dependency = {
            "source_task_id": "task_1",
            "target_task_id": "task_2",
            "type": "invalid_type",
        }

        errors = _validate_dependency_structure(invalid_dependency, 0, task_ids)
        assert "Dependency 0: 'type' must be 'sequential' or 'parallel'" in errors


class TestRoleReferenceValidation:
    """Test role reference validation."""

    def test_valid_role_references(self):
        """Test validation passes for valid role references."""
        task_graph = {
            "tasks": [
                {"id": "task_1", "role": "weather"},
                {"id": "task_2", "role": "timer"},
            ],
            "dependencies": [],
        }
        available_roles = ["weather", "timer", "conversation"]

        errors = _validate_role_references(task_graph, available_roles)
        assert len(errors) == 0

    def test_invalid_role_references(self):
        """Test validation fails for invalid role references."""
        task_graph = {
            "tasks": [
                {"id": "task_1", "role": "nonexistent_role"},
                {"id": "task_2", "role": "another_invalid_role"},
            ],
            "dependencies": [],
        }
        available_roles = ["weather", "timer"]

        errors = _validate_role_references(task_graph, available_roles)
        assert len(errors) == 2
        assert "nonexistent_role" in errors[0]
        assert "another_invalid_role" in errors[1]

    def test_extract_available_role_names(self):
        """Test extraction of role names from formatted text."""
        roles_text = """**weather**: Weather information
  When to use: Get weather data

**timer**: Timer management
  When to use: Set timers

**conversation**: Conversation analysis"""

        role_names = _extract_available_role_names(roles_text)
        assert role_names == ["weather", "timer", "conversation"]


class TestTaskGraphPostProcessing:
    """Test the complete post-processing validation."""

    def test_valid_json_task_graph(self):
        """Test post-processing with valid JSON TaskGraph."""
        valid_task_graph = {
            "tasks": [
                {
                    "id": "task_1",
                    "name": "Check Weather",
                    "description": "Get current weather",
                    "role": "weather",
                    "parameters": {},
                }
            ],
            "dependencies": [],
        }

        llm_result = json.dumps(valid_task_graph)
        pre_data = {"available_roles": "**weather**: Weather information"}

        result = validate_task_graph(llm_result, Mock(), pre_data)

        assert "TaskGraph created successfully" in result
        assert "1 tasks and 0 dependencies" in result
        assert llm_result in result

    def test_invalid_json_handling(self):
        """Test post-processing with invalid JSON."""
        invalid_json = "{ invalid json structure"

        result = validate_task_graph(invalid_json, Mock(), {})

        assert "Invalid JSON generated" in result
        assert "Please try again" in result

    def test_invalid_task_graph_structure_handling(self):
        """Test post-processing with invalid TaskGraph structure."""
        invalid_structure = json.dumps({"invalid": "structure"})

        result = validate_task_graph(invalid_structure, Mock(), {})

        assert "Invalid TaskGraph structure" in result

    def test_invalid_role_references_handling(self):
        """Test post-processing with invalid role references."""
        task_graph_with_invalid_roles = {
            "tasks": [
                {
                    "id": "task_1",
                    "name": "Test",
                    "description": "Test task",
                    "role": "nonexistent_role",
                }
            ],
            "dependencies": [],
        }

        llm_result = json.dumps(task_graph_with_invalid_roles)
        pre_data = {"available_roles": "**weather**: Weather info"}

        result = validate_task_graph(llm_result, Mock(), pre_data)

        assert "Invalid role references" in result
        assert "nonexistent_role" in result

    def test_validation_exception_handling(self):
        """Test post-processing exception handling."""
        # This will cause an exception in validation
        with patch(
            "roles.core_planning.json.loads", side_effect=Exception("Test error")
        ):
            result = validate_task_graph("valid json", Mock(), {})

            assert "Validation error occurred" in result
            assert "Test error" in result


class TestEndToEndPlanning:
    """Test complete planning flow integration."""

    @patch("llm_provider.role_registry.RoleRegistry.get_global_registry")
    def test_complete_planning_flow(self, mock_registry):
        """Test the complete pre-processing -> post-processing flow."""
        # Setup mock registry
        mock_roles = {
            "weather": Mock(
                config={
                    "description": "Weather information",
                    "when_to_use": "Get weather data",
                    "parameters": {},
                }
            ),
            "timer": Mock(
                config={
                    "description": "Timer management",
                    "when_to_use": "Set timers",
                    "parameters": {},
                }
            ),
        }
        mock_registry.return_value.roles = mock_roles

        # Test pre-processing
        context = Mock()
        pre_result = load_available_roles("Plan my morning routine", context, {})

        assert "available_roles" in pre_result
        assert "weather" in pre_result["available_roles"]
        assert "timer" in pre_result["available_roles"]

        # Test post-processing with valid TaskGraph
        valid_task_graph = {
            "tasks": [
                {
                    "id": "task_1",
                    "name": "Check Weather",
                    "description": "Get morning weather",
                    "role": "weather",
                    "parameters": {"location": "current"},
                },
                {
                    "id": "task_2",
                    "name": "Set Alarm",
                    "description": "Set morning alarm",
                    "role": "timer",
                    "parameters": {"duration": "8h", "label": "Wake up"},
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

        llm_result = json.dumps(valid_task_graph)
        post_result = validate_task_graph(llm_result, context, pre_result)

        assert "TaskGraph created successfully" in post_result
        assert "2 tasks and 1 dependencies" in post_result
        assert llm_result in post_result


class TestRoleRegistration:
    """Test role registration function."""

    def test_register_role_structure(self):
        """Test that register_role returns correct structure."""
        registration = register_role()

        assert "config" in registration
        assert "event_handlers" in registration
        assert "tools" in registration
        assert "intents" in registration

        # Config should be the ROLE_CONFIG
        assert registration["config"] == ROLE_CONFIG

        # Should have no event handlers, tools, or intents for new design
        assert registration["event_handlers"] == {}
        assert registration["tools"] == []
        assert registration["intents"] == {}


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_role_list_handling(self):
        """Test handling of empty role list."""
        errors = _validate_role_references({"tasks": []}, [])
        assert len(errors) == 0

    def test_malformed_roles_text_handling(self):
        """Test handling of malformed roles text."""
        malformed_text = "This is not properly formatted role text"
        role_names = _extract_available_role_names(malformed_text)
        assert role_names == []

    def test_none_values_handling(self):
        """Test handling of None values in validation."""
        task_graph = {"tasks": [{"id": "task_1", "role": None}], "dependencies": []}

        errors = _validate_role_references(task_graph, ["weather"])
        # Should not crash, role None should be handled gracefully
        assert len(errors) == 0  # None role is not validated against available roles


# Integration test markers for pytest
pytestmark = pytest.mark.unit
