"""
Tests for the simplified conversation role implementation.

Tests the conversation role's functionality, configuration, and integration
following the timer role NotificationIntent pattern.
"""

from unittest.mock import Mock, patch

import pytest

from common.intents import NotificationIntent
from roles.core_conversation import (
    ROLE_CONFIG,
    ConversationIntent,
    get_conversation_statistics,
    register_role,
    respond_to_user,
    start_new_conversation,
)


class TestConversationRoleConfig:
    """Test conversation role configuration."""

    def test_role_config_structure(self):
        """Test that role config has required structure."""
        assert ROLE_CONFIG["name"] == "conversation"
        assert ROLE_CONFIG["version"] == "3.0.0"
        assert ROLE_CONFIG["llm_type"] == "DEFAULT"
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["memory_enabled"] is True  # Router provides context
        assert ROLE_CONFIG["location_aware"] is False
        assert ROLE_CONFIG["presence_aware"] is False
        assert ROLE_CONFIG["schedule_aware"] is False

    def test_role_config_tools(self):
        """Test that conversation role has simple tools configured."""
        tools_config = ROLE_CONFIG["tools"]
        assert tools_config["automatic"] is True  # Has conversation tools
        assert tools_config["shared"] == []
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
        assert "respond_to_user" in system_prompt
        assert "natural dialogue" in system_prompt


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
        assert (
            len(registration["tools"]) == 2
        )  # respond_to_user + start_new_conversation
        assert ConversationIntent in registration["intents"]

    def test_register_role_no_event_handlers(self):
        """Test no event handlers registered."""
        registration = register_role()
        handlers = registration["event_handlers"]

        assert handlers == {}  # No event handlers


class TestConversationUtilities:
    """Test conversation utility functions."""

    def test_get_conversation_statistics(self):
        """Test conversation statistics function."""
        stats = get_conversation_statistics()

        assert stats["role_name"] == "conversation"
        assert stats["version"] == "3.0.0"
        assert stats["memory_enabled"] is True  # Router provides context
        assert stats["fast_reply"] is True
        assert stats["tools_required"] is True
        assert "recent_memory" in stats["context_types"]
        assert "natural_conversation" in stats["features"]
        assert "notification_intent" in stats["features"]


class TestConversationRoleIntegration:
    """Test conversation role integration with the system."""

    def test_role_config_memory_awareness(self):
        """Test that role is properly configured for memory awareness."""
        assert ROLE_CONFIG["memory_enabled"] is True  # Router provides context
        assert ROLE_CONFIG["location_aware"] is False
        assert ROLE_CONFIG["presence_aware"] is False
        assert ROLE_CONFIG["schedule_aware"] is False

    def test_role_config_fast_reply(self):
        """Test that role is configured for fast reply."""
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["llm_type"] == "DEFAULT"  # Balanced for conversation

    def test_role_config_simple_tools(self):
        """Test that role has simple tools."""
        tools_config = ROLE_CONFIG["tools"]
        assert tools_config["automatic"] is True
        assert tools_config["shared"] == []
        assert tools_config["include_builtin"] is False

    def test_when_to_use_description(self):
        """Test the when_to_use description covers conversation scenarios."""
        when_to_use = ROLE_CONFIG["when_to_use"]
        assert "conversation" in when_to_use.lower()
        assert "dialogue" in when_to_use.lower()
        assert "questions" in when_to_use.lower()


class TestConversationTools:
    """Test conversation tools."""

    def test_respond_to_user(self):
        """Test respond_to_user tool creates NotificationIntent and saves conversation."""
        with patch("roles.core_conversation._save_conversation_exchange"):
            result = respond_to_user(
                "test_user", "Hello", "Hi there! How can I help?", "console"
            )

            assert result == "Hi there! How can I help?"

    def test_start_new_conversation(self):
        """Test starting a new conversation."""
        with patch(
            "roles.core_conversation._archive_current_conversation",
            return_value={"success": True},
        ), patch("roles.core_conversation._initialize_fresh_conversation"):
            result = start_new_conversation("test_user", "Docker Setup", "topic_shift")

            assert result["success"] is True
            assert result["new_topic"] == "Docker Setup"
            assert result["reason"] == "topic_shift"


if __name__ == "__main__":
    pytest.main([__file__])
