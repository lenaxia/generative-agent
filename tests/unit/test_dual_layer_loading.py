"""Tests for dual-layer context loading in roles.

This module tests that roles load both realtime log and assessed memories
in their pre-processing functions.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from common.providers.universal_memory_provider import UniversalMemory


@pytest.fixture
def mock_realtime_messages():
    """Create mock realtime messages."""
    return [
        {
            "id": "msg1",
            "user": "What's the weather?",
            "assistant": "It's sunny today.",
            "role": "conversation",
            "timestamp": time.time() - 100,
            "analyzed": False,
        },
        {
            "id": "msg2",
            "user": "Thanks!",
            "assistant": "You're welcome!",
            "role": "conversation",
            "timestamp": time.time() - 50,
            "analyzed": False,
        },
    ]


@pytest.fixture
def mock_assessed_memories():
    """Create mock assessed memories."""
    return [
        UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="preference",
            content="User prefers morning meetings",
            source_role="conversation",
            timestamp=time.time(),
            summary="User prefers morning meetings",
            importance=0.8,
            tags=["preference", "meetings"],
        ),
        UniversalMemory(
            id="mem2",
            user_id="test_user",
            memory_type="conversation",
            content="User discussed project deadline",
            source_role="conversation",
            timestamp=time.time(),
            summary="Project deadline discussion",
            importance=0.6,
            tags=["project", "deadline"],
        ),
    ]


def test_conversation_loads_both_layers(mock_realtime_messages, mock_assessed_memories):
    """Test conversation role loads both layers."""
    from roles.core_conversation import load_conversation_context

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = mock_realtime_messages

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = mock_assessed_memories
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert "realtime_context" in result
        assert "assessed_memories" in result
        assert "user_id" in result
        # Called twice: once in shared function, once for topics
        assert mock_get_recent.call_count == 2


def test_calendar_loads_both_layers(mock_realtime_messages, mock_assessed_memories):
    """Test calendar role loads both layers."""
    from roles.core_calendar import load_calendar_context

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = mock_realtime_messages

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = mock_assessed_memories
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_calendar_context("test", context, {})

        assert "realtime_context" in result
        assert "assessed_memories" in result
        assert "user_id" in result


def test_planning_loads_both_layers(mock_realtime_messages, mock_assessed_memories):
    """Test planning role loads both layers."""
    from roles.core_planning import load_planning_context

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = mock_realtime_messages

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = mock_assessed_memories
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_planning_context("test", context, {})

        assert "realtime_context" in result
        assert "assessed_memories" in result
        assert "user_id" in result


def test_realtime_context_formatted():
    """Test realtime messages formatted correctly."""
    from roles.core_conversation import load_conversation_context

    messages = [
        {
            "user": "Hello",
            "assistant": "Hi there!",
            "role": "conversation",
            "timestamp": time.time(),
            "analyzed": False,
        }
    ]

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = messages

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = []
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert "Hello" in result["realtime_context"]
        assert "Hi there!" in result["realtime_context"]


def test_assessed_context_formatted():
    """Test assessed memories formatted correctly."""
    from roles.core_conversation import load_conversation_context

    memories = [
        UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="conversation",
            content="Test memory",
            source_role="conversation",
            timestamp=time.time(),
            summary="Test summary",
            importance=0.8,
            tags=["test", "memory"],
        )
    ]

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = memories
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert "Test summary" in result["assessed_memories"]
        assert "test, memory" in result["assessed_memories"]


def test_empty_realtime_handled():
    """Test empty realtime log handled."""
    from roles.core_conversation import load_conversation_context

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = []
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert "No recent messages" in result["realtime_context"]


def test_empty_assessed_handled():
    """Test empty assessed memories handled."""
    from roles.core_conversation import load_conversation_context

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = []
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert "No important memories" in result["assessed_memories"]


def test_importance_filtering():
    """Test only >= 0.7 importance loaded."""
    from roles.core_conversation import load_conversation_context

    memories = [
        UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="conversation",
            content="Important memory",
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
            content="Not important memory",
            source_role="conversation",
            timestamp=time.time(),
            summary="Not important",
            importance=0.5,
            tags=[],
        ),
        UniversalMemory(
            id="mem3",
            user_id="test_user",
            memory_type="conversation",
            content="Very important memory",
            source_role="conversation",
            timestamp=time.time(),
            summary="Very important",
            importance=0.9,
            tags=[],
        ),
    ]

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = []

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = memories
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert "Important" in result["assessed_memories"]
        assert "Very important" in result["assessed_memories"]
        assert "Not important" not in result["assessed_memories"]


def test_message_counts_accurate():
    """Test message counts in context are accurate."""
    from roles.core_conversation import load_conversation_context

    messages = [
        {
            "user": f"Message {i}",
            "assistant": f"Response {i}",
            "role": "conversation",
            "timestamp": time.time(),
            "analyzed": i % 2 == 0,
        }
        for i in range(5)
    ]

    with patch("common.realtime_log.get_recent_messages") as mock_get_recent, patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_get_recent.return_value = messages

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = []
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert result.get("message_count") == 5
        assert (
            result.get("unanalyzed_count") == 2
        )  # 0, 2, 4 are analyzed (True); 1, 3 are not (False)
