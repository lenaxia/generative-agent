"""
Debug test to verify conversation message log fetching and prompt injection.

This test verifies:
1. How messages are loaded from Redis
2. How they're formatted for prompt injection
3. Whether the prompt formatting is working correctly
"""

import time
from unittest.mock import Mock, patch

import pytest

from roles.core_conversation import _load_recent_messages, load_conversation_context


class TestConversationPromptInjection:
    """Debug conversation prompt injection."""

    def test_message_loading_with_real_data(self):
        """Test loading messages with real Redis data format."""

        # Real data from Redis (with "unknown" content issue)
        real_redis_data = [
            {
                "id": "msg_dfc2f78d",
                "timestamp": 1760746379.164114,
                "role": "user",
                "content": "unknown",  # This is the problem!
                "channel_id": "slack:C52L1UK5E",
            },
            {
                "id": "msg_43416fce",
                "timestamp": 1760746379.1641872,
                "role": "assistant",
                "content": "I see you've shared that you're 6'4\" tall! While I can acknowledge your height, I'm here to help - is there something specific you'd like to discuss or ask about?\n",
                "channel_id": "slack:C52L1UK5E",
            },
            {
                "id": "msg_c50cfd7f",
                "timestamp": 1760747713.2657025,
                "role": "user",
                "content": "unknown",  # Should be "im 6'4\""
                "channel_id": "slack:C52L1UK5E",
            },
            {
                "id": "msg_18b45ea5",
                "timestamp": 1760747713.2657106,
                "role": "assistant",
                "content": "I've triggered an analysis of the conversation message where you shared your height of 6'4\". This will help track topics and context from the conversation. Is there anything specific you'd like to know or discuss about your height?\n",
                "channel_id": "slack:C52L1UK5E",
            },
        ]

        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            mock_read.return_value = {"success": True, "value": real_redis_data}

            # Test message loading
            recent_messages = _load_recent_messages("U52L1U8M6", limit=10)

            # Verify we get the data
            assert len(recent_messages) == 4

            # Show the problem: user messages are "unknown"
            user_messages = [msg for msg in recent_messages if msg["role"] == "user"]
            assert len(user_messages) == 2
            assert user_messages[0]["content"] == "unknown"  # This is the problem!
            assert user_messages[1]["content"] == "unknown"  # This is the problem!

            # But assistant messages have the actual content
            assistant_messages = [
                msg for msg in recent_messages if msg["role"] == "assistant"
            ]
            assert len(assistant_messages) == 2
            assert (
                "6'4\"" in assistant_messages[0]["content"]
            )  # Assistant knows the height
            assert "analysis" in assistant_messages[1]["content"]

    def test_prompt_context_formatting(self):
        """Test how conversation context gets formatted for prompt injection."""

        # Mock context with real data pattern
        mock_context = Mock()
        mock_context.user_id = "U52L1U8M6"

        # Mock the Redis data with "unknown" content issue
        real_redis_data = [
            {"role": "user", "content": "unknown", "timestamp": time.time() - 120},
            {
                "role": "assistant",
                "content": "I see you mentioned your height of 6'4\"!",
                "timestamp": time.time() - 60,
            },
            {"role": "user", "content": "unknown", "timestamp": time.time() - 30},
            {
                "role": "assistant",
                "content": "Let me help with that question about your height.",
                "timestamp": time.time() - 10,
            },
        ]

        with (
            patch(
                "roles.core_conversation._load_recent_messages",
                return_value=real_redis_data,
            ),
            patch("roles.core_conversation._load_recent_topics_cache", return_value={}),
            patch("roles.core_conversation._count_unanalyzed_messages", return_value=2),
            patch(
                "roles.core_conversation._extract_current_topics_simple",
                return_value=["height"],
            ),
        ):
            # Test context loading
            result = load_conversation_context("how tall am i?", mock_context, {})

            # Verify context structure
            assert result["user_id"] == "U52L1U8M6"
            assert len(result["recent_messages"]) == 4
            assert result["message_count"] == 4
            assert result["unanalyzed_count"] == 2
            assert "height" in result["current_topics"]

            # Show the issue: recent_messages contains "unknown" for user content
            user_msgs = [
                msg for msg in result["recent_messages"] if msg["role"] == "user"
            ]
            assert len(user_msgs) == 2
            assert all(msg["content"] == "unknown" for msg in user_msgs)

            # But assistant messages have the height information
            assistant_msgs = [
                msg for msg in result["recent_messages"] if msg["role"] == "assistant"
            ]
            assert any("6'4\"" in msg["content"] for msg in assistant_msgs)

    def test_prompt_injection_simulation(self):
        """Simulate how the prompt would look with current data."""

        # This simulates what the system prompt would look like
        system_prompt_template = """You are a conversational AI assistant with topic-based memory.

RECENT CONVERSATION CONTEXT:
{{recent_messages}}

RECENT TOPICS (auto-injected):
{{recent_topics}}

Recent message count: {{message_count}}
Unanalyzed messages: {{unanalyzed_count}}
Current topics: {{current_topics}}"""

        # Mock data that would be injected
        context_data = {
            "recent_messages": [
                {"role": "user", "content": "unknown"},  # Problem: should be "im 6'4\""
                {
                    "role": "assistant",
                    "content": "I see you mentioned your height of 6'4\"!",
                },
                {
                    "role": "user",
                    "content": "unknown",
                },  # Problem: should be "how tall am i?"
            ],
            "recent_topics": {},
            "message_count": 3,
            "unanalyzed_count": 1,
            "current_topics": ["height"],
        }

        # Simulate prompt formatting (what Universal Agent does)
        formatted_prompt = system_prompt_template
        for key, value in context_data.items():
            placeholder = f"{{{{{key}}}}}"  # {{key}}
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

        # Verify the formatted prompt
        assert "unknown" in formatted_prompt  # This shows the problem
        # Check for the height in the assistant's response (escaped in string repr)
        assert "6" in formatted_prompt and "4" in formatted_prompt  # Height is present
        assert "Recent message count: 3" in formatted_prompt

        print("\n=== FORMATTED PROMPT DEBUG ===")
        print(formatted_prompt)
        print("=== END DEBUG ===\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
