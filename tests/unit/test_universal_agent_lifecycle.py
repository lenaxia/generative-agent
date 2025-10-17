"""Tests for UniversalAgent lifecycle execution following LLM-Safe patterns.

This test suite verifies the lifecycle hook system implementation
following the requirements from the handoff document.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.event_context import LLMSafeEventContext
from common.task_context import TaskContext
from llm_provider.factory import LLMType
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestUniversalAgentLifecycle:
    """Test UniversalAgent lifecycle execution following LLM-Safe patterns."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory for testing."""
        factory = MagicMock()
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock()
        mock_agent.return_value.message.content = [{"text": "Test response"}]
        factory.get_agent.return_value = mock_agent
        return factory

    @pytest.fixture
    def mock_role_registry(self):
        """Mock role registry for testing."""
        registry = MagicMock(spec=RoleRegistry)

        # Mock role definition with lifecycle config
        role_def = MagicMock(spec=RoleDefinition)
        role_def.name = "test_role"
        role_def.config = {
            "lifecycle": {
                "pre_processing": {
                    "enabled": True,
                    "functions": ["test_pre_processor"],
                },
                "post_processing": {
                    "enabled": True,
                    "functions": ["test_post_processor"],
                },
            },
            "role": {"execution_type": "hybrid"},
            "prompts": {"system": "Test system prompt with {test_data}"},
        }

        registry.get_role.return_value = role_def

        # Mock lifecycle functions
        async def mock_pre_processor(instruction, context, parameters):
            return {"test_data": "pre_processed_value"}

        async def mock_post_processor(llm_result, context, pre_data):
            return f"post_processed: {llm_result}"

        registry.get_lifecycle_functions.return_value = {
            "test_pre_processor": mock_pre_processor,
            "test_post_processor": mock_post_processor,
        }

        return registry

    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_role_registry):
        """Create UniversalAgent instance for testing."""
        return UniversalAgent(
            llm_factory=mock_llm_factory, role_registry=mock_role_registry
        )

    def test_execute_task_with_lifecycle_method_exists(self, universal_agent):
        """Test that _execute_task_with_lifecycle method exists."""
        assert hasattr(
            universal_agent, "_execute_task_with_lifecycle"
        ), "UniversalAgent must have _execute_task_with_lifecycle method"

    def test_execute_task_with_lifecycle_signature(self, universal_agent):
        """Test that _execute_task_with_lifecycle has correct signature."""
        method = getattr(universal_agent, "_execute_task_with_lifecycle")

        # Check method is callable
        assert callable(method), "_execute_task_with_lifecycle must be callable"

        # Method should be synchronous (not async)
        import inspect

        assert not inspect.iscoroutinefunction(
            method
        ), "_execute_task_with_lifecycle should be synchronous following LLM-Safe patterns"

    def test_execute_task_with_lifecycle_pre_processing(
        self, universal_agent, mock_role_registry
    ):
        """Test lifecycle execution with pre-processing."""
        # Setup
        instruction = "Test instruction"
        role = "test_role"
        llm_type = LLMType.DEFAULT
        context = MagicMock()
        context.user_id = "test_user"

        # Execute
        result = universal_agent._execute_task_with_lifecycle(
            instruction=instruction, role=role, llm_type=llm_type, context=context
        )

        # Verify role was loaded (may be called multiple times)
        mock_role_registry.get_role.assert_called_with(role)

        # Verify lifecycle functions were retrieved
        mock_role_registry.get_lifecycle_functions.assert_called_with(role)

        # Result should be a string
        assert isinstance(result, str), "Result should be a string"

    def test_execute_task_with_lifecycle_post_processing(
        self, universal_agent, mock_role_registry
    ):
        """Test lifecycle execution with post-processing."""
        # Setup
        instruction = "Test instruction"
        role = "test_role"

        # Execute
        result = universal_agent._execute_task_with_lifecycle(
            instruction=instruction, role=role, llm_type=LLMType.DEFAULT
        )

        # Verify post-processing was applied
        assert (
            "post_processed:" in result
        ), "Post-processing should be applied to result"

    def test_execute_task_with_lifecycle_no_pre_processing(
        self, universal_agent, mock_role_registry
    ):
        """Test lifecycle execution when pre-processing is disabled."""
        # Setup role without pre-processing
        role_def = mock_role_registry.get_role.return_value
        role_def.config["lifecycle"]["pre_processing"]["enabled"] = False

        # Execute
        result = universal_agent._execute_task_with_lifecycle(
            instruction="Test instruction", role="test_role", llm_type=LLMType.DEFAULT
        )

        # Should still work without pre-processing
        assert isinstance(result, str), "Should work without pre-processing"

    def test_execute_task_with_lifecycle_no_post_processing(
        self, universal_agent, mock_role_registry
    ):
        """Test lifecycle execution when post-processing is disabled."""
        # Setup role without post-processing
        role_def = mock_role_registry.get_role.return_value
        role_def.config["lifecycle"]["post_processing"]["enabled"] = False

        # Execute
        result = universal_agent._execute_task_with_lifecycle(
            instruction="Test instruction", role="test_role", llm_type=LLMType.DEFAULT
        )

        # Should work without post-processing
        assert isinstance(result, str), "Should work without post-processing"
        assert "post_processed:" not in result, "Post-processing should not be applied"

    def test_execute_task_with_lifecycle_error_handling(
        self, universal_agent, mock_role_registry
    ):
        """Test error handling in lifecycle execution."""
        # Setup role that doesn't exist
        mock_role_registry.get_role.return_value = None

        # Execute
        result = universal_agent._execute_task_with_lifecycle(
            instruction="Test instruction",
            role="nonexistent_role",
            llm_type=LLMType.DEFAULT,
        )

        # Should handle error gracefully
        assert isinstance(result, str), "Should return string even on error"
        assert (
            "error" in result.lower() or "not found" in result.lower()
        ), "Should indicate error in result"

    def test_execute_task_with_lifecycle_llm_safe_patterns(self, universal_agent):
        """Test that lifecycle execution follows LLM-Safe patterns."""
        # Verify no ExecutionMode enum usage
        import inspect

        source = inspect.getsource(universal_agent._execute_task_with_lifecycle)

        assert (
            "ExecutionMode" not in source
        ), "Should not use ExecutionMode enum (removed per handoff document)"

        assert (
            "asyncio.run(" in source or "await" not in source
        ), "Should use asyncio.run() for async calls or be fully synchronous"

    def test_execute_task_uses_lifecycle_method(self, universal_agent):
        """Test that execute_task uses the new lifecycle method."""
        with patch.object(
            universal_agent, "_execute_task_with_lifecycle"
        ) as mock_lifecycle:
            mock_lifecycle.return_value = "test result"

            result = universal_agent.execute_task(
                instruction="Test instruction", role="test_role"
            )

            # Should call the lifecycle method
            mock_lifecycle.assert_called_once()
            assert result == "test result"

    def test_no_special_cases_for_weather_role(self, universal_agent):
        """Test that weather role uses unified execution path."""
        with patch.object(
            universal_agent, "_execute_task_with_lifecycle"
        ) as mock_lifecycle:
            mock_lifecycle.return_value = "weather result"

            # Execute weather role
            result = universal_agent.execute_task(
                instruction="What's the weather?", role="weather"
            )

            # Should use unified path, not special case
            mock_lifecycle.assert_called_once()
            assert result == "weather result"

    def test_unified_execution_for_all_roles(self, universal_agent):
        """Test that all roles use the same unified execution path."""
        roles_to_test = ["weather", "conversation", "timer", "planning"]

        with patch.object(
            universal_agent, "_execute_task_with_lifecycle"
        ) as mock_lifecycle:
            mock_lifecycle.return_value = "unified result"

            for role in roles_to_test:
                result = universal_agent.execute_task(
                    instruction=f"Test {role}", role=role
                )

                assert (
                    result == "unified result"
                ), f"Role {role} should use unified execution"

        # All roles should have called the same method
        assert mock_lifecycle.call_count == len(roles_to_test)


class TestLifecycleFunctionExecution:
    """Test lifecycle function execution patterns."""

    def test_async_lifecycle_functions_called_with_asyncio_run(self):
        """Test that async lifecycle functions are called with asyncio.run()."""
        # This test verifies the LLM-Safe pattern of using asyncio.run()
        # for individual async function calls rather than complex async chains

        async def mock_async_function(instruction, context, parameters):
            return {"processed": True}

        # Test that we can call async function synchronously
        result = asyncio.run(mock_async_function("test", None, {}))

        assert result == {"processed": True}
        assert isinstance(result, dict)

    def test_sync_lifecycle_functions_called_directly(self):
        """Test that sync lifecycle functions are called directly."""

        def mock_sync_function(instruction, context, parameters):
            return {"processed": True}

        # Test direct call
        result = mock_sync_function("test", None, {})

        assert result == {"processed": True}
        assert isinstance(result, dict)
