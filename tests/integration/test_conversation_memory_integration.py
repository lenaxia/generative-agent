"""
Integration test to replicate conversation memory issue.

This test replicates the scenario where:
1. User says "im 6'4""
2. User asks "how tall am i?"
3. Agent should remember the height from previous message
"""

import asyncio
import json
import time
from unittest.mock import Mock, patch

import pytest

from roles.core_conversation import (
    _load_recent_messages,
    _load_recent_topics_cache,
    _save_message_to_global_log,
    load_conversation_context,
    save_message_to_log,
)


class TestConversationMemoryIntegration:
    """Test conversation memory and context integration."""

    @pytest.mark.asyncio
    async def test_conversation_memory_scenario(self):
        """Test the exact scenario from the logs: height memory issue."""

        # Mock context for first message
        mock_context1 = Mock()
        mock_context1.user_id = "U52L1U8M6"
        mock_context1.channel_id = "slack:C52L1UK5E"
        mock_context1.original_prompt = "im 6'4\""

        # Mock Redis to simulate empty initial state
        with (
            patch("roles.shared_tools.redis_tools.redis_read") as mock_read,
            patch("roles.shared_tools.redis_tools.redis_write") as mock_write,
        ):
            # First call - no existing messages
            mock_read.return_value = {"success": False}

            # Test first message processing (user says height)
            result1 = load_conversation_context("im 6'4\"", mock_context1, {})

            assert result1["user_id"] == "U52L1U8M6"
            assert result1["message_count"] == 0  # No previous messages
            assert result1["recent_topics"] == {}  # No cached topics

            # Simulate saving the first message exchange
            save_message_to_log(
                "I've noted your height information.", mock_context1, result1
            )

            # Verify the save was attempted
            mock_write.assert_called()

            # Now simulate the second message - user asks about height
            mock_context2 = Mock()
            mock_context2.user_id = "U52L1U8M6"
            mock_context2.channel_id = "slack:C52L1UK5E"
            mock_context2.original_prompt = "how tall am i?"

            # Mock Redis to return the previous conversation
            saved_messages = [
                {
                    "id": "msg_12345678",
                    "timestamp": time.time() - 60,  # 1 minute ago
                    "role": "user",
                    "content": "im 6'4\"",
                    "channel_id": "slack:C52L1UK5E",
                },
                {
                    "id": "msg_87654321",
                    "timestamp": time.time() - 50,
                    "role": "assistant",
                    "content": "I've noted your height information.",
                    "channel_id": "slack:C52L1UK5E",
                },
            ]

            mock_read.return_value = {
                "success": True,
                "value": saved_messages,
            }

            # Test second message processing (user asks about height)
            result2 = load_conversation_context("how tall am i?", mock_context2, {})

            # The agent should have access to the previous conversation
            assert result2["user_id"] == "U52L1U8M6"
            assert result2["message_count"] == 2  # Previous messages available
            assert len(result2["recent_messages"]) == 2

            # Check that the height information is in recent messages
            user_message = next(
                (msg for msg in result2["recent_messages"] if msg["role"] == "user"),
                None,
            )
            assert user_message is not None
            assert "6'4" in user_message["content"]

    @pytest.mark.asyncio
    async def test_asyncio_event_loop_issue(self):
        """Test the asyncio.run() event loop issue in post-processor."""

        mock_context = Mock()
        mock_context.user_id = "test_user"
        mock_context.channel_id = "test_channel"
        mock_context.original_prompt = "test message"

        # This should NOT use asyncio.run() internally
        with patch("roles.core_conversation._save_message_to_global_log") as mock_save:
            # This should work without asyncio.run() errors
            result = save_message_to_log("test response", mock_context, {})

            assert result == "test response"
            mock_save.assert_called_once_with(
                "test_user", "test message", "test response", "test_channel"
            )

    def test_message_storage_format(self):
        """Test that messages are stored in the correct format for retrieval."""

        with (
            patch("roles.shared_tools.redis_tools.redis_read") as mock_read,
            patch("roles.shared_tools.redis_tools.redis_write") as mock_write,
            patch("uuid.uuid4") as mock_uuid,
        ):
            # Mock existing messages
            existing_messages = [
                {"id": "msg1", "content": "Previous message", "role": "user"}
            ]

            mock_read.return_value = {
                "success": True,
                "value": existing_messages,
            }
            mock_uuid.return_value.hex = "12345678"

            # Save a message exchange
            _save_message_to_global_log(
                "test_user", "im 6'4\"", "I've noted that.", "test_channel"
            )

            # Verify the format includes all necessary fields
            mock_write.assert_called_once()
            call_args = mock_write.call_args[0]
            saved_messages = json.loads(call_args[1])

            # Should have 3 messages: 1 existing + 2 new (user + assistant)
            assert len(saved_messages) == 3

            # Check user message format
            user_msg = saved_messages[-2]
            assert user_msg["role"] == "user"
            assert user_msg["content"] == "im 6'4\""
            assert "id" in user_msg
            assert "timestamp" in user_msg
            assert "channel_id" in user_msg

            # Check assistant message format
            assistant_msg = saved_messages[-1]
            assert assistant_msg["role"] == "assistant"
            assert assistant_msg["content"] == "I've noted that."

    def test_recent_messages_loading(self):
        """Test that recent messages are loaded correctly."""

        # Create test messages with height information
        test_messages = [
            {
                "id": "msg1",
                "timestamp": time.time() - 120,
                "role": "user",
                "content": "hello",
                "channel_id": "test",
            },
            {
                "id": "msg2",
                "timestamp": time.time() - 60,
                "role": "user",
                "content": "im 6'4\"",  # Height information
                "channel_id": "test",
            },
            {
                "id": "msg3",
                "timestamp": time.time() - 30,
                "role": "assistant",
                "content": "I've noted your height.",
                "channel_id": "test",
            },
        ]

        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            mock_read.return_value = {
                "success": True,
                "value": test_messages,
            }

            # Load recent messages
            recent = _load_recent_messages("test_user", limit=10)

            assert len(recent) == 3

            # Find the height message
            height_msg = next((msg for msg in recent if "6'4" in msg["content"]), None)
            assert height_msg is not None
            assert height_msg["role"] == "user"
            assert height_msg["content"] == "im 6'4\""

    def test_empty_cache_handling(self):
        """Test handling of empty recent topics cache."""

        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            # Simulate empty cache
            mock_read.return_value = {"success": False}

            result = _load_recent_topics_cache("test_user")

            assert result == {}
            mock_read.assert_called_once_with("recent_topics_cache:test_user")


if __name__ == "__main__":
    pytest.main([__file__])
