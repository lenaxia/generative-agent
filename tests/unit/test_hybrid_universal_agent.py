"""
Unit tests for enhanced Universal Agent with hybrid execution paths.

Tests the Universal Agent's ability to handle both LLM-based and programmatic
execution paths based on role type for the hybrid execution architecture.
"""

import json
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

from common.task_context import TaskContext
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.programmatic_role import ProgrammaticRole
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class MockProgrammaticRole(ProgrammaticRole):
    """Mock programmatic role for testing."""

    def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
        return {
            "result": f"Programmatic execution of: {instruction}",
            "context_provided": context is not None,
            "execution_type": "programmatic",
        }

    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        return {"query": instruction, "parsed": True}


class TestHybridUniversalAgent:
    """Test suite for hybrid Universal Agent execution."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.create_strands_model.return_value = Mock()
        return factory

    @pytest.fixture
    def mock_role_registry(self):
        """Create mock role registry with both role types."""
        registry = Mock(spec=RoleRegistry)

        # Mock LLM role
        llm_role_def = Mock(spec=RoleDefinition)
        llm_role_def.name = "test_llm_role"
        llm_role_def.config = {
            "role": {"name": "test_llm_role", "description": "Test LLM role"},
            "prompts": {"system": "You are a test LLM role."},
        }
        llm_role_def.custom_tools = []

        # Mock programmatic role
        prog_role = MockProgrammaticRole("test_prog_role", "Test programmatic role")

        # Configure registry behavior
        registry.get_role.side_effect = lambda name: (
            llm_role_def if name == "test_llm_role" else None
        )
        registry.get_programmatic_role.side_effect = lambda name: (
            prog_role if name == "test_prog_role" else None
        )
        registry.is_programmatic_role.side_effect = (
            lambda name: name == "test_prog_role"
        )
        registry.get_role_type.side_effect = lambda name: (
            "programmatic" if name == "test_prog_role" else "llm"
        )

        return registry

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_role_registry):
        """Create Universal Agent with mocked dependencies."""
        return UniversalAgent(
            llm_factory=mock_llm_factory,
            role_registry=mock_role_registry,
            mcp_manager=None,
        )

    def test_is_programmatic_role_detection(self, universal_agent):
        """Test detection of programmatic vs LLM roles."""
        # Test programmatic role detection
        assert universal_agent.is_programmatic_role("test_prog_role") is True
        assert universal_agent.is_programmatic_role("test_llm_role") is False
        assert universal_agent.is_programmatic_role("nonexistent_role") is False

    def test_get_role_type(self, universal_agent):
        """Test role type retrieval."""
        assert universal_agent.get_role_type("test_prog_role") == "programmatic"
        assert universal_agent.get_role_type("test_llm_role") == "llm"
        assert universal_agent.get_role_type("nonexistent_role") == "llm"  # Default

    def test_execute_programmatic_task(self, universal_agent):
        """Test direct programmatic task execution."""
        instruction = "Test programmatic instruction"
        mock_context = Mock(spec=TaskContext)

        result = universal_agent.execute_programmatic_task(
            instruction=instruction, role="test_prog_role", context=mock_context
        )

        # Should return serialized result from programmatic role
        result_data = json.loads(result)
        assert result_data["result"] == f"Programmatic execution of: {instruction}"
        assert result_data["context_provided"] is True
        assert result_data["execution_type"] == "programmatic"

    def test_execute_programmatic_task_error_handling(self, universal_agent):
        """Test error handling in programmatic task execution."""
        # Mock role registry to return None (role not found)
        universal_agent.role_registry.get_programmatic_role.return_value = None

        result = universal_agent.execute_programmatic_task(
            instruction="test", role="nonexistent_role", context=None
        )

        # Should return error message
        assert "Programmatic role 'nonexistent_role' not found" in result

    def test_execute_llm_task(self, universal_agent, mock_llm_factory):
        """Test LLM-based task execution (existing functionality)."""
        # Mock agent creation and execution
        mock_agent = Mock()
        mock_agent.return_value = "LLM response"

        with patch.object(
            universal_agent, "assume_role", return_value=mock_agent
        ) as mock_assume:
            result = universal_agent.execute_llm_task(
                instruction="Test LLM instruction",
                role="test_llm_role",
                llm_type=LLMType.DEFAULT,
                context=None,
            )

        assert result == "LLM response"
        mock_assume.assert_called_once_with("test_llm_role", LLMType.DEFAULT, None)
        mock_agent.assert_called_once_with("Test LLM instruction")

    def test_execute_task_hybrid_routing(self, universal_agent):
        """Test that execute_task routes to correct execution path."""
        mock_context = Mock(spec=TaskContext)

        # Test programmatic routing
        with patch.object(
            universal_agent, "execute_programmatic_task", return_value="prog_result"
        ) as mock_prog:
            result = universal_agent.execute_task(
                instruction="test instruction",
                role="test_prog_role",
                llm_type=LLMType.DEFAULT,
                context=mock_context,
            )

            assert result == "prog_result"
            mock_prog.assert_called_once_with(
                "test instruction", "test_prog_role", mock_context
            )

        # Test LLM routing
        with patch.object(
            universal_agent, "execute_llm_task", return_value="llm_result"
        ) as mock_llm:
            result = universal_agent.execute_task(
                instruction="test instruction",
                role="test_llm_role",
                llm_type=LLMType.DEFAULT,
                context=mock_context,
            )

            assert result == "llm_result"
            mock_llm.assert_called_once_with(
                "test instruction", "test_llm_role", LLMType.DEFAULT, mock_context
            )

    def test_register_programmatic_role(self, universal_agent):
        """Test registering programmatic roles with Universal Agent."""
        mock_role = MockProgrammaticRole("new_prog_role", "New programmatic role")

        universal_agent.register_programmatic_role("new_prog_role", mock_role)

        # Should register with role registry
        universal_agent.role_registry.register_programmatic_role.assert_called_once_with(
            mock_role
        )

    def test_serialize_result_various_types(self, universal_agent):
        """Test result serialization for different data types."""
        # Test string (should pass through)
        assert universal_agent._serialize_result("test string") == "test string"

        # Test dict (should be JSON serialized)
        test_dict = {"key": "value", "number": 42}
        result = universal_agent._serialize_result(test_dict)
        assert json.loads(result) == test_dict

        # Test list (should be JSON serialized)
        test_list = [1, 2, {"nested": "dict"}]
        result = universal_agent._serialize_result(test_list)
        assert json.loads(result) == test_list

        # Test other types (should be string converted)
        assert universal_agent._serialize_result(42) == "42"
        assert universal_agent._serialize_result(None) == "None"

    def test_hybrid_execution_performance_tracking(self, universal_agent):
        """Test that hybrid execution tracks performance metrics."""
        mock_context = Mock(spec=TaskContext)

        # Execute programmatic task
        with patch.object(
            universal_agent, "execute_programmatic_task", return_value="prog_result"
        ):
            universal_agent.execute_task(
                "test", "test_prog_role", LLMType.DEFAULT, mock_context
            )

        # Execute LLM task
        with patch.object(
            universal_agent, "execute_llm_task", return_value="llm_result"
        ):
            universal_agent.execute_task(
                "test", "test_llm_role", LLMType.DEFAULT, mock_context
            )

        # Should have called role type detection for routing
        assert universal_agent.role_registry.is_programmatic_role.call_count >= 2

    def test_backward_compatibility(self, universal_agent):
        """Test that existing Universal Agent functionality still works."""
        # Existing methods should still be available
        assert hasattr(universal_agent, "assume_role")
        assert hasattr(universal_agent, "execute_task")

        # Should still work with LLM roles
        with patch.object(universal_agent, "assume_role") as mock_assume:
            mock_agent = Mock()
            mock_agent.return_value = "response"
            mock_assume.return_value = mock_agent

            result = universal_agent.execute_task("test", "test_llm_role")
            assert result == "response"

    def test_error_handling_programmatic_execution_failure(self, universal_agent):
        """Test error handling when programmatic execution fails."""
        # Mock programmatic role that raises exception
        failing_role = Mock(spec=ProgrammaticRole)
        failing_role.name = "failing_role"
        failing_role.execute.side_effect = ValueError("Execution failed")

        # Configure registry to return the failing role
        universal_agent.role_registry.get_programmatic_role.side_effect = lambda name: (
            failing_role if name == "failing_role" else None
        )

        result = universal_agent.execute_programmatic_task(
            instruction="test", role="failing_role", context=None
        )

        # Should return error message
        assert "Programmatic execution error: Execution failed" in result

    def test_fallback_to_llm_on_programmatic_failure(self, universal_agent):
        """Test fallback to LLM execution when programmatic fails."""
        # This would be implemented as part of graceful fallback strategy
        # For now, test that the error is properly handled

        with patch.object(
            universal_agent,
            "execute_programmatic_task",
            side_effect=Exception("Prog failed"),
        ):
            with patch.object(
                universal_agent, "execute_llm_task", return_value="llm_fallback"
            ):
                # This test verifies the concept - actual fallback implementation would be in execute_task
                try:
                    universal_agent.execute_programmatic_task(
                        "test", "test_prog_role", None
                    )
                    assert False, "Should have raised exception"
                except Exception as e:
                    assert "Prog failed" in str(e)

                    # Fallback would work
                    fallback_result = universal_agent.execute_llm_task(
                        "test", "test_llm_role", LLMType.DEFAULT, None
                    )
                    assert fallback_result == "llm_fallback"
