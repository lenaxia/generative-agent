"""
Tests for the simplified topic-based conversation role implementation.

Tests the conversation role's functionality with global message log and
LLM-triggered analysis approach.
"""

import json
import time
from unittest.mock import Mock, patch

import pytest

from common.intents import NotificationIntent
from roles.core_conversation import (
    ROLE_CONFIG,
    TopicAnalysisIntent,
    TopicSearchIntent,
    _count_unanalyzed_messages,
    _extract_current_topics_simple,
    _get_unanalyzed_messages,
    _load_recent_messages,
    _load_recent_topics_cache,
    _save_message_to_global_log,
    _search_topics_with_relevance,
    _update_analysis_pointer,
    _update_recent_topics_cache,
    analyze_conversation,
    get_conversation_statistics,
    load_conversation_context,
    process_topic_analysis_intent,
    process_topic_search_intent,
    register_role,
    save_message_to_log,
    search_topics,
)


class TestConversationRoleConfig:
    """Test conversation role configuration."""

    def test_role_config_structure(self):
        """Test that role config has required structure."""
        assert ROLE_CONFIG["name"] == "conversation"
        assert ROLE_CONFIG["version"] == "5.0.0"
        assert ROLE_CONFIG["llm_type"] == "DEFAULT"
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["memory_enabled"] is True
        assert ROLE_CONFIG["location_aware"] is False
        assert ROLE_CONFIG["presence_aware"] is False
        assert ROLE_CONFIG["schedule_aware"] is False

    def test_role_config_tools(self):
        """Test that conversation role has tools configured."""
        tools_config = ROLE_CONFIG["tools"]
        assert tools_config["automatic"] is True
        assert "redis_tools" in tools_config["shared"]
        assert tools_config["include_builtin"] is False
        assert tools_config["fast_reply"]["enabled"] is True

    def test_role_config_lifecycle(self):
        """Test lifecycle configuration."""
        lifecycle = ROLE_CONFIG["lifecycle"]
        assert lifecycle["pre_processing"]["enabled"] is True
        assert "load_conversation_context" in lifecycle["pre_processing"]["functions"]
        assert lifecycle["post_processing"]["enabled"] is True
        assert "save_message_to_log" in lifecycle["post_processing"]["functions"]

    def test_role_config_prompts(self):
        """Test system prompt configuration."""
        system_prompt = ROLE_CONFIG["prompts"]["system"]
        assert "conversational AI" in system_prompt
        assert "analyze_conversation" in system_prompt
        assert "topic-based memory" in system_prompt
        assert "{{recent_messages}}" in system_prompt
        assert "{{recent_topics}}" in system_prompt
        assert "{{unanalyzed_count}}" in system_prompt


class TestTopicAnalysisIntent:
    """Test topic analysis intent."""

    def test_topic_analysis_intent_creation(self):
        """Test creating topic analysis intent."""
        intent = TopicAnalysisIntent(
            user_id="test_user", analysis_trigger="llm_triggered"
        )

        assert intent.user_id == "test_user"
        assert intent.analysis_trigger == "llm_triggered"
        assert intent.validate() is True

    def test_topic_analysis_intent_validation(self):
        """Test topic analysis intent validation."""
        # Valid intent
        valid_intent = TopicAnalysisIntent(user_id="test_user")
        assert valid_intent.validate() is True

        # Invalid intent - empty user_id
        invalid_intent = TopicAnalysisIntent(user_id="")
        assert invalid_intent.validate() is False

        # Invalid intent - whitespace only user_id
        invalid_intent2 = TopicAnalysisIntent(user_id="   ")
        assert invalid_intent2.validate() is False


class TestTopicSearchIntent:
    """Test topic search intent."""

    def test_topic_search_intent_creation(self):
        """Test creating topic search intent."""
        intent = TopicSearchIntent(
            user_id="test_user", query="dogs and pets", relevance_threshold=0.8
        )

        assert intent.user_id == "test_user"
        assert intent.query == "dogs and pets"
        assert intent.relevance_threshold == 0.8
        assert intent.validate() is True

    def test_topic_search_intent_validation(self):
        """Test topic search intent validation."""
        # Valid intent
        valid_intent = TopicSearchIntent(user_id="test_user", query="test query")
        assert valid_intent.validate() is True

        # Invalid intent - empty user_id
        invalid_intent1 = TopicSearchIntent(user_id="", query="test")
        assert invalid_intent1.validate() is False

        # Invalid intent - empty query
        invalid_intent2 = TopicSearchIntent(user_id="test_user", query="")
        assert invalid_intent2.validate() is False

        # Invalid intent - bad threshold
        invalid_intent3 = TopicSearchIntent(
            user_id="test_user", query="test", relevance_threshold=1.5
        )
        assert invalid_intent3.validate() is False


class TestConversationTools:
    """Test conversation tools."""

    def test_analyze_conversation_tool(self):
        """Test analyze_conversation tool."""
        result = analyze_conversation()

        assert result["success"] is True
        assert "analysis triggered" in result["message"]
        assert result["intent"]["type"] == "TopicAnalysisIntent"
        assert result["intent"]["analysis_trigger"] == "llm_triggered"

    def test_search_topics_tool(self):
        """Test search_topics tool."""
        result = search_topics("dogs and training")

        assert result["success"] is True
        assert "Searching for topics" in result["message"]
        assert result["query"] == "dogs and training"
        assert result["intent"]["type"] == "TopicSearchIntent"
        assert result["intent"]["query"] == "dogs and training"
        assert result["intent"]["relevance_threshold"] == 0.8


class TestLifecycleFunctions:
    """Test lifecycle functions."""

    @pytest.mark.asyncio
    async def test_load_conversation_context(self):
        """Test loading conversation context."""
        mock_context = Mock()
        mock_context.user_id = "test_user"

        with (
            patch(
                "roles.core_conversation._load_recent_messages",
                return_value=[
                    {"content": "Hello", "role": "user"},
                    {"content": "Hi there!", "role": "assistant"},
                ],
            ),
            patch(
                "roles.core_conversation._load_recent_topics_cache",
                return_value={
                    "greetings": {
                        "summary": "User greetings",
                        "key_details": ["polite"],
                        "relevance_score": 0.9,
                    }
                },
            ),
            patch("roles.core_conversation._count_unanalyzed_messages", return_value=5),
            patch(
                "roles.core_conversation._extract_current_topics_simple",
                return_value=["general"],
            ),
        ):
            result = load_conversation_context("Hello again", mock_context, {})

            assert result["user_id"] == "test_user"
            assert len(result["recent_messages"]) == 2
            assert "greetings" in result["recent_topics"]
            assert result["message_count"] == 2
            assert result["unanalyzed_count"] == 5
            assert "general" in result["current_topics"]

    @pytest.mark.asyncio
    async def test_save_message_to_log(self):
        """Test saving message to log."""
        mock_context = Mock()
        mock_context.user_id = "test_user"
        mock_context.channel_id = "general"
        mock_context.original_prompt = "Hello"

        with patch("roles.core_conversation._save_message_to_global_log") as mock_save:
            result = save_message_to_log("Hi there!", mock_context, {})

            assert result == "Hi there!"
            mock_save.assert_called_once_with(
                "test_user", "Hello", "Hi there!", "general"
            )


class TestHelperFunctions:
    """Test helper functions."""

    def test_load_recent_messages(self):
        """Test loading recent messages."""
        mock_messages = [
            {"id": "msg1", "content": "Hello", "role": "user"},
            {"id": "msg2", "content": "Hi!", "role": "assistant"},
        ]

        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            mock_read.return_value = {
                "success": True,
                "value": mock_messages,
            }

            result = _load_recent_messages("test_user", limit=10)

            assert len(result) == 2
            assert result[0]["content"] == "Hello"
            mock_read.assert_called_once_with("conversation:messages:test_user")

    def test_load_recent_topics_cache(self):
        """Test loading recent topics cache."""
        mock_cached_topics = {
            "dogs": {
                "summary": "User interested in dogs",
                "key_details": ["Golden Retriever", "family-friendly"],
                "relevance_score": 0.9,
            }
        }

        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            mock_read.return_value = {
                "success": True,
                "value": mock_cached_topics,
            }

            result = _load_recent_topics_cache("test_user")

            assert "dogs" in result
            assert result["dogs"]["relevance_score"] == 0.9
            mock_read.assert_called_once_with(
                "conversation:recent_topics_cache:test_user"
            )

    def test_search_topics_with_relevance(self):
        """Test searching topics with relevance scoring."""
        mock_topics = {
            "dogs": {
                "summary": "User interested in Golden Retriever dogs",
                "key_details": ["Golden Retriever", "family-friendly"],
                "importance": 8,
            },
            "cats": {
                "summary": "User has indoor cats",
                "key_details": ["indoor cats", "two cats"],
                "importance": 6,
            },
        }

        with (
            patch("roles.shared_tools.redis_tools.redis_read") as mock_read,
            patch("roles.core_conversation._update_recent_topics_cache") as mock_cache,
        ):
            mock_read.return_value = {"success": True, "value": mock_topics}

            result = _search_topics_with_relevance("test_user", "dogs", threshold=0.8)

            assert "dogs" in result
            assert "cats" not in result  # Below relevance threshold
            assert result["dogs"]["relevance_score"] >= 0.8
            mock_cache.assert_called_once()

    def test_update_recent_topics_cache(self):
        """Test updating recent topics cache with TTL."""
        topics = {"dogs": {"summary": "test", "relevance_score": 0.9}}

        with patch("roles.shared_tools.redis_tools.redis_write") as mock_write:
            _update_recent_topics_cache("test_user", topics)

            mock_write.assert_called_once_with(
                "recent_topics_cache:test_user", json.dumps(topics), ttl=3600
            )

    def test_extract_current_topics_simple(self):
        """Test simple topic extraction."""
        messages = [
            {"content": "I want to get a dog", "role": "user"},
            {"content": "What about college applications?", "role": "user"},
            {"content": "My job interview went well", "role": "user"},
        ]

        result = _extract_current_topics_simple(messages)

        assert "pets" in result
        assert "education" in result
        assert "career" in result

    def test_count_unanalyzed_messages(self):
        """Test counting unanalyzed messages."""
        mock_analysis_data = {
            "last_message_index": 5,
            "last_analysis_timestamp": time.time(),
        }

        mock_messages = [{"id": f"msg{i}"} for i in range(10)]  # 10 total messages

        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            mock_read.side_effect = [
                {
                    "success": True,
                    "value": mock_analysis_data,
                },  # analysis data
                {"success": True, "value": mock_messages},  # messages
            ]

            result = _count_unanalyzed_messages("test_user")

            assert result == 5  # 10 total - 5 analyzed = 5 unanalyzed

    def test_get_unanalyzed_messages(self):
        """Test getting unanalyzed messages."""
        mock_analysis_data = {
            "last_message_index": 2,
            "last_analysis_timestamp": time.time(),
        }

        mock_messages = [
            {"id": "msg1", "content": "Old message 1"},
            {"id": "msg2", "content": "Old message 2"},
            {"id": "msg3", "content": "New message 1"},
            {"id": "msg4", "content": "New message 2"},
        ]

        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            mock_read.side_effect = [
                {
                    "success": True,
                    "value": mock_analysis_data,
                },  # analysis data
                {"success": True, "value": mock_messages},  # messages
            ]

            result = _get_unanalyzed_messages("test_user")

            assert len(result) == 2  # Messages after index 2
            assert result[0]["content"] == "New message 1"
            assert result[1]["content"] == "New message 2"

    def test_update_analysis_pointer(self):
        """Test updating analysis pointer."""
        mock_messages = [{"id": f"msg{i}"} for i in range(10)]

        with (
            patch("roles.shared_tools.redis_tools.redis_read") as mock_read,
            patch("roles.shared_tools.redis_tools.redis_write") as mock_write,
        ):
            mock_read.return_value = {
                "success": True,
                "value": mock_messages,
            }

            _update_analysis_pointer("test_user", 5)

            # Should have called write with updated pointer
            mock_write.assert_called_once()
            call_args = mock_write.call_args[0]
            assert call_args[0] == "conversation:last_analysis:test_user"

            saved_pointer = json.loads(call_args[1])
            assert saved_pointer["last_message_index"] == 10  # Total message count
            assert saved_pointer["analyzed_message_count"] == 5

    def test_save_message_to_global_log(self):
        """Test saving message to global log."""
        existing_messages = [
            {"id": "msg1", "content": "Previous message", "role": "user"}
        ]

        with (
            patch("roles.shared_tools.redis_tools.redis_read") as mock_read,
            patch("roles.shared_tools.redis_tools.redis_write") as mock_write,
            patch("uuid.uuid4") as mock_uuid,
        ):
            mock_read.return_value = {
                "success": True,
                "value": existing_messages,
            }
            mock_uuid.return_value.hex = "12345678"

            _save_message_to_global_log("test_user", "Hello", "Hi there!", "general")

            # Should have called write with updated messages
            mock_write.assert_called_once()
            call_args = mock_write.call_args[0]
            assert call_args[0] == "conversation:messages:test_user"

            saved_messages = json.loads(call_args[1])
            assert len(saved_messages) == 3  # 1 existing + 2 new (user + assistant)
            assert saved_messages[-2]["content"] == "Hello"
            assert saved_messages[-1]["content"] == "Hi there!"


class TestRoleRegistration:
    """Test role registration functionality."""

    def test_register_role_structure(self):
        """Test role registration returns correct structure."""
        registration = register_role()

        assert "config" in registration
        assert "event_handlers" in registration
        assert "tools" in registration
        assert "intents" in registration
        assert "pre_processors" in registration
        assert "post_processors" in registration

        assert registration["config"] == ROLE_CONFIG
        assert len(registration["tools"]) == 2  # analyze_conversation and search_topics
        assert TopicAnalysisIntent in registration["intents"]
        assert (
            registration["intents"][TopicAnalysisIntent]
            == process_topic_analysis_intent
        )

    def test_register_role_no_event_handlers(self):
        """Test no event handlers registered."""
        registration = register_role()
        handlers = registration["event_handlers"]

        assert handlers == {}  # No event handlers needed

    def test_register_role_tools(self):
        """Test both analyze_conversation and search_topics tools are registered."""
        registration = register_role()
        tools = registration["tools"]

        assert len(tools) == 2
        tool_names = [tool.__name__ for tool in tools]
        assert "analyze_conversation" in tool_names
        assert "search_topics" in tool_names


class TestConversationUtilities:
    """Test conversation utility functions."""

    def test_get_conversation_statistics(self):
        """Test conversation statistics function."""
        stats = get_conversation_statistics()

        assert stats["role_name"] == "conversation"
        assert stats["version"] == "5.0.0"
        assert stats["architecture"] == "global_message_log_with_topic_analysis"
        assert "global_message_log" in stats["features"]
        assert "topic_extraction" in stats["features"]
        assert "memory_importance_ranking" in stats["features"]
        assert "llm_triggered_analysis" in stats["features"]
        assert "analysis_pointer_tracking" in stats["features"]
        assert "natural_conversation" in stats["features"]


class TestConversationRoleIntegration:
    """Test conversation role integration with the system."""

    def test_role_config_memory_awareness(self):
        """Test that role is properly configured for memory awareness."""
        assert ROLE_CONFIG["memory_enabled"] is True
        assert ROLE_CONFIG["location_aware"] is False
        assert ROLE_CONFIG["presence_aware"] is False
        assert ROLE_CONFIG["schedule_aware"] is False

    def test_role_config_fast_reply(self):
        """Test that role is configured for fast reply."""
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["llm_type"] == "DEFAULT"

    def test_when_to_use_description(self):
        """Test the when_to_use description covers conversation scenarios."""
        when_to_use = ROLE_CONFIG["when_to_use"]
        assert "conversation" in when_to_use.lower()
        assert "dialogue" in when_to_use.lower()
        assert "questions" in when_to_use.lower()

    def test_simplified_architecture(self):
        """Test that the architecture is properly simplified."""
        # Two tools now
        registration = register_role()
        assert len(registration["tools"]) == 2

        # Two intent types now
        assert len(registration["intents"]) == 2

        # No event handlers
        assert len(registration["event_handlers"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
