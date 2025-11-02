"""Tests for universal realtime log."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_redis():
    """Mock Redis operations."""
    with patch("common.realtime_log.redis_zadd") as mock_zadd, patch(
        "common.realtime_log.redis_zrevrange"
    ) as mock_zrevrange, patch(
        "common.realtime_log.redis_zremrangebyscore"
    ) as mock_zremrangebyscore:
        mock_zadd.return_value = {"success": True}
        mock_zrevrange.return_value = {"success": True, "values": []}
        mock_zremrangebyscore.return_value = {"success": True}
        yield {
            "zadd": mock_zadd,
            "zrevrange": mock_zrevrange,
            "zremrangebyscore": mock_zremrangebyscore,
        }


class TestRealtimeLogBasics:
    """Test basic realtime log operations."""

    def test_add_message_to_realtime_log(self, mock_redis):
        """Test adding a message to the realtime log."""
        from common.realtime_log import add_message

        result = add_message(
            user_id="user123",
            user_message="Test user message",
            assistant_response="Test assistant response",
            role="conversation",
        )

        assert result is True
        mock_redis["zadd"].assert_called_once()

        # Verify the message structure
        call_args = mock_redis["zadd"].call_args
        assert call_args[0][0] == "realtime:user123"
        # Score should be timestamp
        assert isinstance(call_args[0][1], float)
        # Value should be JSON string
        message_data = json.loads(call_args[0][2])
        assert message_data["user"] == "Test user message"
        assert message_data["assistant"] == "Test assistant response"
        assert message_data["role"] == "conversation"
        assert message_data["analyzed"] is False

    def test_add_message_with_metadata(self, mock_redis):
        """Test adding message with additional metadata."""
        from common.realtime_log import add_message

        result = add_message(
            user_id="user123",
            user_message="Test",
            assistant_response="Response",
            role="calendar",
            metadata={"event_id": "evt-123"},
        )

        assert result is True
        call_args = mock_redis["zadd"].call_args
        message_data = json.loads(call_args[0][2])
        assert message_data["metadata"] == {"event_id": "evt-123"}

    def test_add_message_failure(self, mock_redis):
        """Test handling Redis failure."""
        from common.realtime_log import add_message

        mock_redis["zadd"].return_value = {"success": False, "error": "Redis error"}

        result = add_message(
            user_id="user123",
            user_message="Test",
            assistant_response="Response",
            role="conversation",
        )

        assert result is False


class TestRealtimeLogRetrieval:
    """Test retrieving messages from realtime log."""

    def test_get_recent_messages(self, mock_redis):
        """Test getting recent messages."""
        from common.realtime_log import get_recent_messages

        # Setup mock data
        messages = [
            json.dumps(
                {
                    "id": f"msg-{i}",
                    "user": f"User message {i}",
                    "assistant": f"Assistant response {i}",
                    "role": "conversation",
                    "timestamp": time.time() - i,
                    "analyzed": False,
                }
            )
            for i in range(5)
        ]

        mock_redis["zrevrange"].return_value = {"success": True, "values": messages}

        result = get_recent_messages(user_id="user123", limit=5)

        assert len(result) == 5
        assert result[0]["user"] == "User message 0"
        mock_redis["zrevrange"].assert_called_once_with("realtime:user123", 0, 4)

    def test_get_recent_messages_with_limit(self, mock_redis):
        """Test getting messages with custom limit."""
        from common.realtime_log import get_recent_messages

        mock_redis["zrevrange"].return_value = {"success": True, "values": []}

        get_recent_messages(user_id="user123", limit=10)

        mock_redis["zrevrange"].assert_called_once_with("realtime:user123", 0, 9)

    def test_get_recent_messages_empty(self, mock_redis):
        """Test getting messages when log is empty."""
        from common.realtime_log import get_recent_messages

        mock_redis["zrevrange"].return_value = {"success": True, "values": []}

        result = get_recent_messages(user_id="user123")

        assert len(result) == 0


class TestUnanalyzedMessages:
    """Test tracking and retrieving unanalyzed messages."""

    def test_get_unanalyzed_messages(self, mock_redis):
        """Test getting only unanalyzed messages."""
        from common.realtime_log import get_unanalyzed_messages

        messages = [
            json.dumps(
                {
                    "id": "msg-1",
                    "user": "Message 1",
                    "assistant": "Response 1",
                    "role": "conversation",
                    "analyzed": False,
                }
            ),
            json.dumps(
                {
                    "id": "msg-2",
                    "user": "Message 2",
                    "assistant": "Response 2",
                    "role": "conversation",
                    "analyzed": True,
                }
            ),
            json.dumps(
                {
                    "id": "msg-3",
                    "user": "Message 3",
                    "assistant": "Response 3",
                    "role": "conversation",
                    "analyzed": False,
                }
            ),
        ]

        mock_redis["zrevrange"].return_value = {"success": True, "values": messages}

        result = get_unanalyzed_messages(user_id="user123")

        assert len(result) == 2
        assert result[0]["id"] == "msg-1"
        assert result[1]["id"] == "msg-3"

    def test_mark_as_analyzed(self, mock_redis):
        """Test marking messages as analyzed."""
        from common.realtime_log import mark_as_analyzed

        # Setup mock - return messages to update
        messages = [
            json.dumps(
                {
                    "id": "msg-1",
                    "user": "Test",
                    "assistant": "Response",
                    "analyzed": False,
                }
            )
        ]

        mock_redis["zrevrange"].return_value = {"success": True, "values": messages}

        result = mark_as_analyzed(user_id="user123", message_ids=["msg-1"])

        assert result is True
        # Should have removed and re-added with analyzed=True
        assert mock_redis["zadd"].called


class TestRealtimeLogCleanup:
    """Test automatic cleanup of old messages."""

    def test_cleanup_old_messages(self, mock_redis):
        """Test removing messages older than 24 hours."""
        from common.realtime_log import cleanup_old_messages

        result = cleanup_old_messages(user_id="user123", max_age_hours=24)

        assert result is True
        mock_redis["zremrangebyscore"].assert_called_once()

        # Verify cutoff calculation
        call_args = mock_redis["zremrangebyscore"].call_args
        assert call_args[0][0] == "realtime:user123"
        assert call_args[0][1] == "-inf"
        # Score should be (now - 24 hours)
        cutoff = call_args[0][2]
        expected_cutoff = time.time() - (24 * 60 * 60)
        assert abs(cutoff - expected_cutoff) < 1  # Within 1 second
