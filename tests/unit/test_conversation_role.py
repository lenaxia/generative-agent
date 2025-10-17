"""
Tests for the conversation role implementation.

Tests the conversation role's functionality, configuration, and integration
with the Universal Agent System's context-aware architecture.
"""

import time
from unittest.mock import Mock, patch

import pytest

from common.event_context import LLMSafeEventContext
from common.intents import AuditIntent, NotificationIntent
from roles.core_conversation import (
    ROLE_CONFIG,
    ConversationIntent,
    archive_conversation,
    get_conversation_statistics,
    handle_conversation_end,
    handle_conversation_start,
    load_conversation,
    register_role,
    save_message,
    search_archive,
    start_new_conversation,
)


class TestConversationRoleConfig:
    """Test conversation role configuration."""

    def test_role_config_structure(self):
        """Test that role config has required structure."""
        assert ROLE_CONFIG["name"] == "conversation"
        assert ROLE_CONFIG["version"] == "2.0.0"
        assert ROLE_CONFIG["llm_type"] == "DEFAULT"
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["memory_enabled"] is False  # Manages its own state
        assert ROLE_CONFIG["location_aware"] is False
        assert ROLE_CONFIG["presence_aware"] is False
        assert ROLE_CONFIG["schedule_aware"] is False

    def test_role_config_tools(self):
        """Test that conversation role has conversation management tools configured."""
        tools_config = ROLE_CONFIG["tools"]
        assert tools_config["automatic"] is True
        assert "redis_tools" in tools_config["shared"]
        assert tools_config["include_builtin"] is False
        assert tools_config["fast_reply"]["enabled"] is True

    def test_role_config_parameters(self):
        """Test role parameter schema."""
        params = ROLE_CONFIG["parameters"]
        # No specific parameters needed - LLM handles conversation naturally
        assert isinstance(params, dict)

    def test_role_config_prompts(self):
        """Test system prompt configuration."""
        system_prompt = ROLE_CONFIG["prompts"]["system"]
        assert "conversational AI" in system_prompt
        assert "persistent memory" in system_prompt
        assert "topic management" in system_prompt


class TestConversationIntent:
    """Test conversation-specific intent."""

    def test_conversation_intent_creation(self):
        """Test creating conversation intent."""
        intent = ConversationIntent(
            interaction_type="conversation", user_message="How does AI work?"
        )

        assert intent.interaction_type == "conversation"
        assert intent.user_message == "How does AI work?"
        assert intent.validate() is True

    def test_conversation_intent_validation(self):
        """Test conversation intent validation."""
        # Valid intent (default interaction_type)
        valid_intent = ConversationIntent()
        assert valid_intent.validate() is True

        # Valid intent with custom interaction_type
        valid_intent2 = ConversationIntent(interaction_type="chat")
        assert valid_intent2.validate() is True

    def test_conversation_intent_optional_fields(self):
        """Test conversation intent with optional fields."""
        intent = ConversationIntent()
        assert intent.interaction_type == "conversation"  # default value
        assert intent.user_message is None
        assert intent.validate() is True


class TestEventHandlers:
    """Test conversation event handlers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_context = Mock(spec=LLMSafeEventContext)
        self.mock_context.get_safe_channel.return_value = "test_channel"
        self.mock_context.user_id = "test_user"

    def test_handle_conversation_start_with_dict(self):
        """Test conversation start handler with dictionary event data."""
        event_data = {
            "type": "question",
            "topic": "technology",
            "message": "How does AI work?",
        }

        intents = handle_conversation_start(event_data, self.mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], AuditIntent)
        assert intents[0].action == "conversation_started"
        assert (
            intents[0].details["message"] == "How does AI work?"
        )  # Gets message from dict
        assert intents[0].user_id == "test_user"

    def test_handle_conversation_start_with_string(self):
        """Test conversation start handler with string event data."""
        event_data = "Hello, how are you?"

        intents = handle_conversation_start(event_data, self.mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], AuditIntent)
        assert intents[0].action == "conversation_started"
        assert intents[0].details["message"] == event_data

    def test_handle_conversation_start_success(self):
        """Test successful conversation start handler."""
        # Test that the handler works correctly under normal conditions
        normal_context = Mock(spec=LLMSafeEventContext)
        normal_context.get_safe_channel.return_value = "test_channel"
        normal_context.user_id = "test_user"

        intents = handle_conversation_start({}, normal_context)

        assert len(intents) == 1
        assert isinstance(intents[0], AuditIntent)
        assert intents[0].action == "conversation_started"

    def test_handle_conversation_end(self):
        """Test conversation end handler."""
        event_data = {"duration": 300, "message_count": 5}

        intents = handle_conversation_end(event_data, self.mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], AuditIntent)
        assert intents[0].action == "conversation_ended"
        assert "channel" in intents[0].details
        assert "timestamp" in intents[0].details


class TestRoleRegistration:
    """Test role registration functionality."""

    def test_register_role_structure(self):
        """Test role registration returns correct structure."""
        registration = register_role()

        assert "config" in registration
        assert "event_handlers" in registration
        assert "tools" in registration
        assert "intents" in registration

        assert registration["config"] == ROLE_CONFIG
        assert len(registration["tools"]) == 5  # Conversation management tools
        assert ConversationIntent in registration["intents"]

    def test_register_role_event_handlers(self):
        """Test registered event handlers."""
        registration = register_role()
        handlers = registration["event_handlers"]

        assert "CONVERSATION_START" in handlers
        assert "CONVERSATION_END" in handlers
        assert handlers["CONVERSATION_START"] == handle_conversation_start
        assert handlers["CONVERSATION_END"] == handle_conversation_end


class TestConversationUtilities:
    """Test conversation utility functions."""

    def test_get_conversation_statistics(self):
        """Test conversation statistics function."""
        stats = get_conversation_statistics()

        assert stats["role_name"] == "conversation"
        assert stats["version"] == "2.0.0"
        assert stats["memory_enabled"] is False  # Manages its own state
        assert stats["fast_reply"] is True
        assert stats["tools_required"] is True
        assert "conversation_history" in stats["context_types"]
        assert "persistent_conversations" in stats["features"]
        assert "topic_detection" in stats["features"]
        assert "auto_archiving" in stats["features"]


class TestConversationRoleIntegration:
    """Test conversation role integration with the system."""

    def test_role_config_memory_awareness(self):
        """Test that role is properly configured for memory awareness."""
        assert ROLE_CONFIG["memory_enabled"] is False  # Manages its own state
        assert ROLE_CONFIG["location_aware"] is False
        assert ROLE_CONFIG["presence_aware"] is False
        assert ROLE_CONFIG["schedule_aware"] is False

    def test_role_config_fast_reply(self):
        """Test that role is configured for fast reply."""
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["llm_type"] == "DEFAULT"  # Balanced for conversation

    def test_role_config_has_tools(self):
        """Test that role has conversation management tools."""
        tools_config = ROLE_CONFIG["tools"]
        assert tools_config["automatic"] is True
        assert "redis_tools" in tools_config["shared"]
        assert tools_config["include_builtin"] is False

    def test_when_to_use_description(self):
        """Test the when_to_use description covers conversation scenarios."""
        when_to_use = ROLE_CONFIG["when_to_use"]
        assert "conversation" in when_to_use.lower()
        assert "dialogue" in when_to_use.lower()
        assert "questions" in when_to_use.lower()
        assert "follow-up" in when_to_use.lower()


class TestConversationTools:
    """Test conversation management tools."""

    def test_load_conversation_empty(self):
        """Test loading conversation when none exists."""
        with patch(
            "roles.shared_tools.redis_tools.redis_read", return_value={"success": False}
        ):
            result = load_conversation("test_user")

            assert result["messages"] == []
            assert result["topic"] is None
            assert result["message_count"] == 0

    def test_save_message_new_conversation(self):
        """Test saving message to new conversation."""
        with patch(
            "roles.shared_tools.redis_tools.redis_read", return_value={"success": False}
        ), patch("roles.shared_tools.redis_tools.redis_write") as mock_write:
            result = save_message("test_user", "user", "Hello", "console")

            assert result["success"] is True
            assert result["message_count"] == 1
            mock_write.assert_called_once()

    def test_start_new_conversation(self):
        """Test starting a new conversation."""
        with patch(
            "roles.shared_tools.redis_tools.redis_read", return_value={"success": False}
        ), patch("roles.shared_tools.redis_tools.redis_write") as mock_write:
            result = start_new_conversation("test_user", "Docker Setup", "topic_shift")

            assert result["success"] is True
            assert result["new_topic"] == "Docker Setup"
            assert result["reason"] == "topic_shift"
            mock_write.assert_called_once()

    def test_search_archive_empty(self):
        """Test searching archive when none exists."""
        with patch(
            "roles.shared_tools.redis_tools.redis_read", return_value={"success": False}
        ):
            result = search_archive("test_user", "docker")

            assert result == []

    def test_archive_conversation_too_short(self):
        """Test archiving conversation that's too short."""
        short_conversation = {
            "messages": [{"content": "hi"}, {"content": "hello"}],
            "topic": "greeting",
        }

        with patch(
            "roles.core_conversation.load_conversation", return_value=short_conversation
        ):
            result = archive_conversation("test_user")

            assert result["success"] is False
            assert result["success"] is False


if __name__ == "__main__":
    pytest.main([__file__])
