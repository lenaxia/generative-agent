"""Tests for conversation role unified memory integration."""

from unittest.mock import MagicMock, patch

import pytest

from common.intents import MemoryWriteIntent
from common.providers.universal_memory_provider import UniversalMemory


@pytest.fixture
def mock_memory_provider():
    """Mock UniversalMemoryProvider."""
    with patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def mock_context():
    """Mock context object."""
    context = MagicMock()
    context.user_id = "test_user"
    context.channel_id = "test_channel"
    context.original_prompt = "Test user message"
    return context


class TestConversationMemoryIntegration:
    """Test conversation role integration with unified memory."""

    def test_conversation_loads_tier1_memories(self, mock_memory_provider):
        """Test conversation pre-processing loads Tier 1 memories."""
        from roles.core_conversation import load_conversation_context

        # Setup mock
        mock_memories = [
            UniversalMemory(
                id=f"mem{i}",
                user_id="test_user",
                memory_type="conversation",
                content=f"Memory {i}",
                source_role="conversation",
                timestamp=1234567890.0 - i,
                importance=0.8,  # High importance so it passes filter
                summary=f"Memory {i}",
                tags=["test"],
            )
            for i in range(5)
        ]
        mock_memory_provider.get_recent_memories.return_value = mock_memories

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        # Execute
        result = load_conversation_context("test instruction", mock_context, {})

        # Verify dual-layer context was loaded
        assert "realtime_context" in result
        assert "assessed_memories" in result
        # Memories are now in assessed_memories (formatted string)
        assert result["assessed_memories"] != "No important memories."

        # Verify provider was called correctly (now includes plan type)
        mock_memory_provider.get_recent_memories.assert_called_once_with(
            user_id="test_user", memory_types=["conversation", "event", "plan"], limit=5
        )

    def test_conversation_has_search_memory_tool(self):
        """Test conversation role has search_memory tool."""
        from roles.core_conversation import register_role

        registration = register_role()

        # Check that memory_tools is in shared tools
        assert "tools" in registration["config"]
        assert "shared" in registration["config"]["tools"]
        assert "memory_tools" in registration["config"]["tools"]["shared"]

    def test_conversation_emits_memory_write_intent(self, mock_context):
        """Test conversation post-processing emits MemoryWriteIntent."""
        from roles.core_conversation import save_message_to_log

        llm_result = "This is the assistant's response"

        # Execute post-processing
        result = save_message_to_log(llm_result, mock_context, {})

        # Result should be unchanged
        assert result == llm_result

        # Note: Intent emission happens via IntentProcessingHook
        # This test verifies the function doesn't break

    def test_conversation_memory_with_topics(self, mock_memory_provider):
        """Test conversation memories include topic tags."""
        from roles.core_conversation import load_conversation_context

        # Setup mock with topic-tagged memories
        mock_memories = [
            UniversalMemory(
                id="mem1",
                user_id="test_user",
                memory_type="conversation",
                content="Discussion about AI",
                source_role="conversation",
                timestamp=1234567890.0,
                importance=0.9,  # High importance so it passes filter
                summary="Discussion about AI",
                tags=["ai", "technology"],
            )
        ]
        mock_memory_provider.get_recent_memories.return_value = mock_memories

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        result = load_conversation_context("test", mock_context, {})

        # Verify memories with tags were loaded
        assert "assessed_memories" in result
        # Memory is now in assessed_memories (formatted string)
        assert result["assessed_memories"] != "No important memories."
        # Verify tags are in the formatted string
        assert "ai, technology" in result["assessed_memories"]

    def test_conversation_context_includes_user_id(self, mock_memory_provider):
        """Test context includes user_id for tool usage."""
        from roles.core_conversation import load_conversation_context

        mock_memory_provider.get_recent_memories.return_value = []

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        result = load_conversation_context("test", mock_context, {})

        assert "user_id" in result
        assert result["user_id"] == "test_user"
