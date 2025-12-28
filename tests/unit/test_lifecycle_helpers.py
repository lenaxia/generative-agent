"""Tests for shared lifecycle helper functions.

This module tests the shared lifecycle functions used by multiple roles
for loading dual-layer context and saving to realtime log.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from common.providers.universal_memory_provider import UniversalMemory
from roles.shared_tools.lifecycle_helpers import (
    load_dual_layer_context,
    save_to_realtime_log,
)


@pytest.fixture
def mock_context():
    """Create mock context."""
    context = MagicMock()
    context.user_id = "test_user"
    return context


@pytest.fixture
def mock_realtime_messages():
    """Create mock realtime messages."""
    return [
        {
            "user": "Hello",
            "assistant": "Hi!",
            "role": "conversation",
            "timestamp": time.time(),
            "analyzed": False,
        }
    ]


@pytest.fixture
def mock_assessed_memories():
    """Create mock assessed memories."""
    return [
        UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="conversation",
            content="Important memory",
            source_role="conversation",
            timestamp=time.time(),
            summary="Important memory",
            importance=0.8,
            tags=["important"],
        )
    ]


def test_load_dual_layer_context_success(
    mock_context, mock_realtime_messages, mock_assessed_memories
):
    """Test successful dual-layer context loading."""
    with (
        patch("common.realtime_log.get_recent_messages") as mock_get_recent,
        patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class,
    ):
        mock_get_recent.return_value = mock_realtime_messages

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = mock_assessed_memories
        mock_provider_class.return_value = mock_provider

        result = load_dual_layer_context(mock_context)

        assert "realtime_context" in result
        assert "assessed_memories" in result
        assert "user_id" in result
        assert result["user_id"] == "test_user"
        assert "Hello" in result["realtime_context"]
        assert "Important memory" in result["assessed_memories"]


def test_load_dual_layer_context_with_memory_types(
    mock_context, mock_assessed_memories
):
    """Test loading with specific memory types."""
    with (
        patch("common.realtime_log.get_recent_messages") as mock_get_recent,
        patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class,
    ):
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = mock_assessed_memories
        mock_provider_class.return_value = mock_provider

        result = load_dual_layer_context(
            mock_context, memory_types=["conversation", "event"]
        )

        mock_provider.get_recent_memories.assert_called_once()
        call_kwargs = mock_provider.get_recent_memories.call_args[1]
        assert call_kwargs["memory_types"] == ["conversation", "event"]


def test_load_dual_layer_context_with_limits(mock_context):
    """Test loading with custom limits."""
    with (
        patch("common.realtime_log.get_recent_messages") as mock_get_recent,
        patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class,
    ):
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = []
        mock_provider_class.return_value = mock_provider

        result = load_dual_layer_context(
            mock_context, realtime_limit=20, assessed_limit=15
        )

        mock_get_recent.assert_called_once_with("test_user", limit=20)
        call_kwargs = mock_provider.get_recent_memories.call_args[1]
        assert call_kwargs["limit"] == 15


def test_load_dual_layer_context_filters_importance(mock_context):
    """Test importance filtering."""
    memories = [
        UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="conversation",
            content="Important",
            source_role="conversation",
            timestamp=time.time(),
            summary="Important",
            importance=0.8,
            tags=[],
        ),
        UniversalMemory(
            id="mem2",
            user_id="test_user",
            memory_type="conversation",
            content="Not important",
            source_role="conversation",
            timestamp=time.time(),
            summary="Not important",
            importance=0.5,
            tags=[],
        ),
    ]

    with (
        patch("common.realtime_log.get_recent_messages") as mock_get_recent,
        patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class,
    ):
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = memories
        mock_provider_class.return_value = mock_provider

        result = load_dual_layer_context(mock_context, importance_threshold=0.7)

        assert "Important" in result["assessed_memories"]
        assert "Not important" not in result["assessed_memories"]


def test_load_dual_layer_context_empty_data(mock_context):
    """Test handling empty data."""
    with (
        patch("common.realtime_log.get_recent_messages") as mock_get_recent,
        patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class,
    ):
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = []
        mock_provider_class.return_value = mock_provider

        result = load_dual_layer_context(mock_context)

        assert "No recent messages" in result["realtime_context"]
        assert "No important memories" in result["assessed_memories"]


def test_save_to_realtime_log_success(mock_context):
    """Test successful save to realtime log."""
    # Set original_prompt to None explicitly
    mock_context.original_prompt = None

    with patch("common.realtime_log.add_message") as mock_add:
        mock_add.return_value = True

        pre_data = {"_instruction": "Test user message"}
        result = save_to_realtime_log(
            "Test response", mock_context, pre_data, "conversation"
        )

        assert result == "Test response"
        mock_add.assert_called_once()
        call_kwargs = mock_add.call_args[1]
        assert call_kwargs["user_id"] == "test_user"
        assert call_kwargs["user_message"] == "Test user message"
        assert call_kwargs["assistant_response"] == "Test response"
        assert call_kwargs["role"] == "conversation"


def test_save_to_realtime_log_from_context(mock_context):
    """Test save using original_prompt from context."""
    mock_context.original_prompt = "From context"

    with patch("common.realtime_log.add_message") as mock_add:
        mock_add.return_value = True

        pre_data = {}
        result = save_to_realtime_log("Response", mock_context, pre_data, "calendar")

        call_kwargs = mock_add.call_args[1]
        assert call_kwargs["user_message"] == "From context"


def test_save_to_realtime_log_fallback_to_unknown(mock_context):
    """Test fallback to unknown when no prompt available."""
    # Set original_prompt to None explicitly
    mock_context.original_prompt = None

    with patch("common.realtime_log.add_message") as mock_add:
        mock_add.return_value = True

        pre_data = {}
        result = save_to_realtime_log("Response", mock_context, pre_data, "planning")

        call_kwargs = mock_add.call_args[1]
        assert call_kwargs["user_message"] == "unknown"


def test_save_to_realtime_log_handles_non_string_result(mock_context):
    """Test handling non-string LLM results."""
    with patch("common.realtime_log.add_message") as mock_add:
        mock_add.return_value = True

        pre_data = {"_instruction": "Test"}
        result = save_to_realtime_log(
            {"result": "object"}, mock_context, pre_data, "planning"
        )

        call_kwargs = mock_add.call_args[1]
        assert isinstance(call_kwargs["assistant_response"], str)
