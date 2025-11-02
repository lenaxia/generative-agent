"""
Tests for Single-File Timer Role - LLM-Safe Architecture

Tests the new LLM-friendly single-file timer role that follows Documents 25 & 26
architecture patterns with heartbeat-driven timer monitoring.

Key architectural principles tested:
- Single event loop compliance (no asyncio.sleep())
- Intent-based processing (pure functions returning intents)
- Heartbeat-driven timer monitoring (Redis polling every 5 seconds)
- LLM-safe patterns (predictable, simple, self-contained)

Following TDD principles - tests written first.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from common.intents import AuditIntent, Intent, NotificationIntent


class TestSingleFileTimerRoleLLMSafe:
    """Test single-file timer role LLM-safe architecture."""

    def test_timer_role_config_structure(self):
        """Test that timer role has proper LLM-safe config structure."""
        from roles.core_timer import ROLE_CONFIG

        # Verify LLM-safe config structure
        assert "name" in ROLE_CONFIG
        assert ROLE_CONFIG["name"] == "timer"
        assert "version" in ROLE_CONFIG
        assert (
            ROLE_CONFIG["version"] == "6.0.0"
        )  # New version for LLM-safe architecture
        assert "description" in ROLE_CONFIG
        assert "heartbeat-driven" in ROLE_CONFIG["description"]
        assert "llm_type" in ROLE_CONFIG
        assert ROLE_CONFIG["llm_type"] == "WEAK"
        assert "fast_reply" in ROLE_CONFIG
        assert "when_to_use" in ROLE_CONFIG
        assert "timer" in ROLE_CONFIG["when_to_use"].lower()

        # Verify tools configuration excludes built-ins (Document 26 pattern)
        assert ROLE_CONFIG["tools"]["include_builtin"] is False
        assert "redis_tools" in ROLE_CONFIG["tools"]["shared"]

    def test_timer_intent_definitions(self):
        """Test timer-specific intent definitions follow Document 25 patterns."""
        from roles.core_timer import (
            TimerCancellationIntent,
            TimerCreationIntent,
            TimerExpiryIntent,
            TimerListingIntent,
        )

        # Test TimerCreationIntent
        creation_intent = TimerCreationIntent(
            timer_id="test_123",
            duration="5m",
            duration_seconds=300,
            label="Test timer",
        )
        assert creation_intent.validate() is True
        assert creation_intent.timer_id == "test_123"
        assert creation_intent.duration_seconds == 300
        assert creation_intent.label == "Test timer"

        # Test TimerCancellationIntent
        cancel_intent = TimerCancellationIntent(timer_id="test_456")
        assert cancel_intent.validate() is True
        assert cancel_intent.timer_id == "test_456"

        # Test TimerListingIntent
        list_intent = TimerListingIntent()
        assert list_intent.validate() is True  # No required parameters

        # Test TimerExpiryIntent
        expiry_intent = TimerExpiryIntent(
            timer_id="test_789", original_duration="10m", label="Expired timer"
        )
        assert expiry_intent.validate() is True
        assert expiry_intent.timer_id == "test_789"
        assert expiry_intent.original_duration == "10m"

    def test_timer_intent_validation(self):
        """Test timer intent validation follows Document 25 patterns."""
        from roles.core_timer import TimerCreationIntent

        # Valid intent
        valid_intent = TimerCreationIntent(
            timer_id="test_valid", duration="1m", duration_seconds=60
        )
        assert valid_intent.validate() is True

        # Invalid intent (empty timer_id)
        invalid_intent = TimerCreationIntent(
            timer_id="", duration="1m", duration_seconds=60
        )
        assert invalid_intent.validate() is False

        # Invalid intent (zero duration)
        invalid_duration_intent = TimerCreationIntent(
            timer_id="test_invalid", duration="0m", duration_seconds=0
        )
        assert invalid_duration_intent.validate() is False

    def test_heartbeat_monitoring_handler(self):
        """Test heartbeat monitoring handler follows Document 25 pure function pattern."""
        from roles.core_timer import handle_heartbeat_monitoring

        # Create mock context
        mock_context = type("MockContext", (), {"user_id": "user123"})()

        # Mock Redis operations to return expired timers
        with patch("roles.core_timer._get_expired_timers_from_redis") as mock_redis:
            with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
                # Setup mock to return expired timer
                mock_redis.return_value = ["timer_123"]
                mock_read.return_value = {
                    "success": True,
                    "data": {
                        "duration": "5m",
                        "label": "Test timer",
                        "user_id": "user123",
                        "channel_id": "channel456",
                    },
                }

                # Call handler
                intents = handle_heartbeat_monitoring({}, mock_context)

                # Should return list of TimerExpiryIntents
                assert isinstance(intents, list)
                if intents:  # Only check if timers were found
                    from roles.core_timer import TimerExpiryIntent

                    assert all(
                        isinstance(intent, TimerExpiryIntent) for intent in intents
                    )
                    assert all(intent.validate() for intent in intents)

    def test_timer_expiry_handler_pure_function(self):
        """Test timer expiry handler is pure function returning intents."""
        from roles.core_timer import handle_timer_expiry

        # Create mock context
        mock_context = type(
            "MockContext",
            (),
            {
                "user_id": "user123",
                "get_safe_channel": lambda self: "channel456",
            },
        )()

        # Call handler with valid data
        intents = handle_timer_expiry(
            ["timer_123", "Set timer for 5 minutes"], mock_context
        )

        # Should return list of intents
        assert isinstance(intents, list)
        assert len(intents) >= 1

        # Should contain notification intent
        notification_intents = [i for i in intents if isinstance(i, NotificationIntent)]
        assert len(notification_intents) >= 1

        # Should contain audit intent
        audit_intents = [i for i in intents if isinstance(i, AuditIntent)]
        assert len(audit_intents) >= 1

        # All intents should be valid
        assert all(intent.validate() for intent in intents)

        # Test error handling
        error_intents = handle_timer_expiry(None, mock_context)
        assert isinstance(error_intents, list)
        assert len(error_intents) >= 1
        # Should contain error notification
        error_notifications = [
            i
            for i in error_intents
            if isinstance(i, NotificationIntent) and "error" in i.message.lower()
        ]
        assert len(error_notifications) >= 1

    def test_timer_tools_declarative_pattern(self):
        """Test timer tools follow Document 25 declarative pattern."""
        from roles.core_timer import cancel_timer, list_timers, set_timer

        # Test set_timer tool - should return intent data, no side effects
        result = set_timer("5m", "Test timer")
        assert result["success"] is True
        assert "message" in result
        assert "Timer set" in result["message"]
        assert "intent" in result
        assert result["intent"]["type"] == "TimerCreationIntent"
        assert result["intent"]["duration"] == "5m"
        assert result["intent"]["label"] == "Test timer"

        # Test cancel_timer tool - should return intent data
        result = cancel_timer("timer_123")
        assert result["success"] is True
        assert "cancelled" in result["message"]
        assert "intent" in result
        assert result["intent"]["type"] == "TimerCancellationIntent"
        assert result["intent"]["timer_id"] == "timer_123"

        # Test list_timers tool - should return intent data
        result = list_timers()
        assert result["success"] is True
        assert "message" in result
        assert "intent" in result
        assert result["intent"]["type"] == "TimerListingIntent"

    def test_timer_helper_functions_llm_safe(self):
        """Test timer helper functions are LLM-safe and predictable."""
        from roles.core_timer import _parse_duration, _parse_timer_event_data

        # Test event data parsing - should handle various formats
        timer_id, request = _parse_timer_event_data(["timer_123", "Test request"])
        assert timer_id == "timer_123"
        assert request == "Test request"

        # Test dict parsing
        timer_id, request = _parse_timer_event_data(
            {"timer_id": "timer_456", "original_request": "Dict request"}
        )
        assert timer_id == "timer_456"
        assert request == "Dict request"

        # Test error handling
        timer_id, request = _parse_timer_event_data("invalid_data")
        assert "unknown" in timer_id or "parse_error" in timer_id

        # Test duration parsing - should handle common formats
        assert _parse_duration("5m") == 300
        assert _parse_duration("1h") == 3600
        assert _parse_duration("30s") == 30
        assert _parse_duration("120") == 120

        # Test error handling
        with pytest.raises(ValueError):
            _parse_duration("invalid_duration")

    def test_role_registration_llm_safe_structure(self):
        """Test role registration follows Document 25/26 patterns."""
        from roles.core_timer import ROLE_CONFIG, register_role

        role_info = register_role()

        # Should have all required sections from Document 25
        assert "config" in role_info
        assert "event_handlers" in role_info
        assert "tools" in role_info
        assert "intents" in role_info

        # Config should match ROLE_CONFIG
        assert role_info["config"] == ROLE_CONFIG

        # Should have heartbeat event handler (Document 25 pattern)
        assert "FAST_HEARTBEAT_TICK" in role_info["event_handlers"]
        assert "TIMER_EXPIRED" in role_info["event_handlers"]

        # Should have tools
        assert len(role_info["tools"]) == 3  # set_timer, cancel_timer, list_timers

        # Should have intent handlers
        assert len(role_info["intents"]) >= 4  # All timer intents

    def test_no_asyncio_sleep_usage(self):
        """Test that the role doesn't use asyncio.sleep() (Document 25 compliance)."""
        import inspect

        import roles.core_timer as timer_role

        # Get all functions and methods in the module
        functions = [
            obj
            for name, obj in inspect.getmembers(timer_role)
            if inspect.isfunction(obj) or inspect.ismethod(obj)
        ]

        # Check source code for asyncio.sleep usage
        for func in functions:
            try:
                source = inspect.getsource(func)
                assert (
                    "asyncio.sleep" not in source
                ), f"Function {func.__name__} uses asyncio.sleep"
            except OSError:
                # Built-in functions don't have source
                pass

    def test_redis_sorted_set_integration(self):
        """Test Redis sorted set integration for efficient timer queuing."""
        from roles.core_timer import _get_expired_timers_from_redis

        # Mock Redis client
        with patch(
            "roles.shared_tools.redis_tools._get_redis_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Setup mock to return expired timers
            mock_client.zrangebyscore.return_value = [b"timer_123", b"timer_456"]

            # Test getting expired timers
            current_time = int(time.time())
            expired_timers = _get_expired_timers_from_redis(current_time)

            # Should query Redis sorted set correctly
            mock_client.zrangebyscore.assert_called_once_with(
                "timer:active_queue", 0, current_time
            )

            # Should remove expired timers from queue
            mock_client.zremrangebyscore.assert_called_once_with(
                "timer:active_queue", 0, current_time
            )

            # Should return decoded timer IDs
            assert expired_timers == ["timer_123", "timer_456"]


class TestTimerRoleIntentProcessing:
    """Test timer role intent processing integration."""

    @pytest.mark.asyncio
    async def test_timer_creation_intent_processing(self):
        """Test timer creation intent processing with Redis operations."""
        from roles.core_timer import TimerCreationIntent, process_timer_creation_intent

        # Create test intent
        intent = TimerCreationIntent(
            timer_id="test_timer",
            duration="5m",
            duration_seconds=300,
            label="Test timer",
            user_id="user123",
            channel_id="channel456",
        )

        # Mock Redis operations
        with patch("roles.shared_tools.redis_tools.redis_write") as mock_write:
            with patch(
                "roles.shared_tools.redis_tools._get_redis_client"
            ) as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                mock_write.return_value = {"success": True}

                # Process intent
                await process_timer_creation_intent(intent)

                # Should store timer metadata
                mock_write.assert_called_once()
                call_args = mock_write.call_args
                assert call_args[0][0] == "timer:data:test_timer"

                # Should add to sorted set
                mock_client.zadd.assert_called_once()
                zadd_args = mock_client.zadd.call_args
                assert zadd_args[0][0] == "timer:active_queue"
                assert "test_timer" in zadd_args[0][1]

    @pytest.mark.asyncio
    async def test_timer_cancellation_intent_processing(self):
        """Test timer cancellation intent processing with Redis operations."""
        from roles.core_timer import (
            TimerCancellationIntent,
            process_timer_cancellation_intent,
        )

        # Create test intent
        intent = TimerCancellationIntent(timer_id="test_timer", user_id="user123")

        # Mock Redis operations
        with patch("roles.shared_tools.redis_tools.redis_read") as mock_read:
            with patch("roles.shared_tools.redis_tools.redis_delete") as mock_delete:
                with patch(
                    "roles.shared_tools.redis_tools._get_redis_client"
                ) as mock_get_client:
                    mock_client = MagicMock()
                    mock_get_client.return_value = mock_client
                    mock_read.return_value = {
                        "success": True,
                        "value": {
                            "user_id": "user123",
                            "event_context": {
                                "user_id": "user123",
                                "channel_id": "test",
                            },
                        },
                    }
                    mock_delete.return_value = {"success": True}

                    # Process intent
                    await process_timer_cancellation_intent(intent)

                    # Should remove from sorted set
                    mock_client.zrem.assert_called_once_with(
                        "timer:active_queue", "test_timer"
                    )

                    # Should delete timer metadata
                    mock_delete.assert_called_once_with("timer:data:test_timer")

    def test_timer_role_llm_safety_patterns(self):
        """Test that timer role follows LLM-safe patterns from Documents 25/26."""
        from roles.core_timer import handle_heartbeat_monitoring

        # Create mock context
        mock_context = type("MockContext", (), {"user_id": "user123"})()

        # Handler should be pure function (no side effects)
        with patch("roles.core_timer._get_expired_timers_from_redis") as mock_redis:
            mock_redis.return_value = []

            # Multiple calls should return consistent results
            result1 = handle_heartbeat_monitoring({}, mock_context)
            result2 = handle_heartbeat_monitoring({}, mock_context)

            # Results should be consistent (same structure)
            assert type(result1) == type(result2)
            assert isinstance(result1, list)
            assert isinstance(result2, list)
            assert all(isinstance(i, Intent) for i in result1)
            assert all(isinstance(i, Intent) for i in result2)

    def test_single_file_architecture_compliance(self):
        """Test that timer role follows single-file architecture from Document 26."""
        import roles.core_timer as timer_role

        # Should have all required components in single file
        assert hasattr(timer_role, "ROLE_CONFIG")
        assert hasattr(timer_role, "TimerCreationIntent")
        assert hasattr(timer_role, "TimerCancellationIntent")
        assert hasattr(timer_role, "TimerListingIntent")
        assert hasattr(timer_role, "TimerExpiryIntent")
        assert hasattr(timer_role, "handle_timer_expiry")
        assert hasattr(timer_role, "handle_heartbeat_monitoring")
        assert hasattr(timer_role, "set_timer")
        assert hasattr(timer_role, "cancel_timer")
        assert hasattr(timer_role, "list_timers")
        assert hasattr(timer_role, "register_role")

        # Should have intent processors
        assert hasattr(timer_role, "process_timer_creation_intent")
        assert hasattr(timer_role, "process_timer_cancellation_intent")
        assert hasattr(timer_role, "process_timer_listing_intent")
        assert hasattr(timer_role, "process_timer_expiry_intent")


if __name__ == "__main__":
    pytest.main([__file__])
