"""Test timer TaskContext method call fix.

Tests that the timer lifecycle functions correctly use TaskContext.get_metadata()
instead of the non-existent TaskContext.get() method.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.task_context import TaskContext
from common.task_graph import TaskGraph
from roles.timer.lifecycle import parse_timer_parameters


class TestTimerTaskContextFix:
    """Test timer TaskContext method call fix."""

    @pytest.fixture
    def mock_task_context(self):
        """Create a mock TaskContext with proper methods."""
        # Create a real TaskContext instance
        task_graph = TaskGraph(tasks=[], dependencies=[])
        context = TaskContext(
            task_graph=task_graph,
            context_id="test_context",
            user_id="U52L1U8M6",
            channel_id="C52L1UK5E",
        )

        # Add some metadata
        context.set_metadata("original_user_input", "set timer 20s")
        context.set_metadata("device_context", {"type": "mobile"})
        context.set_metadata("source", "slack")

        return context

    @pytest.fixture
    def mock_timer_manager(self):
        """Create a mock timer manager."""
        manager = AsyncMock()
        manager.create_timer.return_value = "timer_12345"
        return manager

    @patch("roles.timer.lifecycle.get_timer_manager")
    @pytest.mark.asyncio
    async def test_parse_timer_parameters_with_taskcontext(
        self, mock_get_manager, mock_task_context, mock_timer_manager
    ):
        """Test that parse_timer_parameters works with TaskContext using get_metadata."""
        mock_get_manager.return_value = mock_timer_manager

        # Test parameters
        instruction = "set timer 20s"
        parameters = {"action": "set", "duration": "20s"}

        # Call the function that was previously failing
        result = await parse_timer_parameters(
            instruction=instruction, context=mock_task_context, parameters=parameters
        )

        # Verify the function completed successfully
        assert result is not None
        assert result.get("action_requested") == "set"
        assert result.get("duration_seconds") == 20
        assert "execution_result" in result

        # Verify the execution was successful
        execution_result = result["execution_result"]
        assert execution_result.get("success") is True
        assert "timer_id" in execution_result

        # Verify timer manager was called with correct parameters
        mock_timer_manager.create_timer.assert_called_once()
        call_args = mock_timer_manager.create_timer.call_args

        # Check that user_id and channel_id were extracted correctly
        assert call_args.kwargs["user_id"] == "U52L1U8M6"
        assert call_args.kwargs["channel_id"] == "slack:C52L1UK5E"

        # Check that request_context was built correctly using get_metadata
        request_context = call_args.kwargs["request_context"]
        assert request_context["original_request"] == "set timer 20s"
        assert request_context["execution_context"]["user_id"] == "U52L1U8M6"
        assert request_context["execution_context"]["channel"] == "slack:C52L1UK5E"
        assert request_context["execution_context"]["device_context"] == {
            "type": "mobile"
        }
        assert request_context["execution_context"]["source"] == "slack"

    @patch("roles.timer.lifecycle.get_timer_manager")
    @pytest.mark.asyncio
    async def test_parse_timer_parameters_without_context(
        self, mock_get_manager, mock_timer_manager
    ):
        """Test that parse_timer_parameters works when context is None."""
        mock_get_manager.return_value = mock_timer_manager

        # Test parameters
        instruction = "set timer 30s"
        parameters = {"action": "set", "duration": "30s"}

        # Call the function with no context
        result = await parse_timer_parameters(
            instruction=instruction, context=None, parameters=parameters
        )

        # Verify the function completed successfully
        assert result is not None
        assert result.get("action_requested") == "set"
        assert result.get("duration_seconds") == 30
        assert "execution_result" in result

        # Verify the execution was successful with default values
        execution_result = result["execution_result"]
        assert execution_result.get("success") is True

        # Verify timer manager was called with default values
        mock_timer_manager.create_timer.assert_called_once()
        call_args = mock_timer_manager.create_timer.call_args

        # Check that default values were used
        assert call_args.kwargs["user_id"] == "system"
        assert call_args.kwargs["channel_id"] == "default"

        # Check that request_context was built with empty defaults
        request_context = call_args.kwargs["request_context"]
        assert request_context["original_request"] == ""
        assert request_context["execution_context"]["device_context"] == {}
        assert request_context["execution_context"]["source"] == "unknown"

    @patch("roles.timer.lifecycle.get_timer_manager")
    @pytest.mark.asyncio
    async def test_parse_timer_parameters_list_action(
        self, mock_get_manager, mock_task_context, mock_timer_manager
    ):
        """Test that parse_timer_parameters works for list action with TaskContext."""
        mock_get_manager.return_value = mock_timer_manager
        mock_timer_manager.list_timers.return_value = [
            {"id": "timer_1", "name": "Test Timer 1"},
            {"id": "timer_2", "name": "Test Timer 2"},
        ]

        # Add user_id to metadata for list operation
        mock_task_context.set_metadata("user_id", "U52L1U8M6")

        # Test parameters
        instruction = "list timers"
        parameters = {"action": "list"}

        # Call the function
        result = await parse_timer_parameters(
            instruction=instruction, context=mock_task_context, parameters=parameters
        )

        # Verify the function completed successfully
        assert result is not None
        assert result.get("action_requested") == "list"

        # Verify the execution was successful
        execution_result = result["execution_result"]
        assert execution_result.get("success") is True
        assert execution_result.get("count") == 2

        # Verify list_timers was called with correct user_id from metadata
        mock_timer_manager.list_timers.assert_called_once_with(user_id="U52L1U8M6")

    def test_taskcontext_has_get_metadata_method(self):
        """Verify that TaskContext has get_metadata method but not get method."""
        task_graph = TaskGraph(tasks=[], dependencies=[])
        context = TaskContext(task_graph=task_graph)

        # Verify get_metadata exists
        assert hasattr(context, "get_metadata")
        assert callable(getattr(context, "get_metadata"))

        # Verify get method does not exist
        assert not hasattr(context, "get")

        # Test get_metadata works
        context.set_metadata("test_key", "test_value")
        assert context.get_metadata("test_key") == "test_value"
        assert context.get_metadata("nonexistent_key", "default") == "default"
