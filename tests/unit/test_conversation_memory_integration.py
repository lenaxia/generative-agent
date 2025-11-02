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
            )
            for i in range(5)
        ]
        mock_memory_provider.get_recent_memories.return_value = mock_memories

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        # Execute
        result = load_conversation_context("test instruction", mock_context, {})

        # Verify Tier 1 memories were loaded
        assert "tier1_memories" in result
        assert len(result["tier1_memories"]) == 5

        # Verify provider was called correctly
        mock_memory_provider.get_recent_memories.assert_called_once_with(
            user_id="test_user", memory_types=["conversation", "event"], limit=5
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
                tags=["ai", "technology"],
            )
        ]
        mock_memory_provider.get_recent_memories.return_value = mock_memories

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        result = load_conversation_context("test", mock_context, {})

        # Verify memories with tags were loaded
        assert "tier1_memories" in result
        assert len(result["tier1_memories"]) == 1
        assert result["tier1_memories"][0].tags == ["ai", "technology"]

    def test_conversation_context_includes_user_id(self, mock_memory_provider):
        """Test context includes user_id for tool usage."""
        from roles.core_conversation import load_conversation_context

        mock_memory_provider.get_recent_memories.return_value = []

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        result = load_conversation_context("test", mock_context, {})

        assert "user_id" in result
        assert result["user_id"] == "test_user"
