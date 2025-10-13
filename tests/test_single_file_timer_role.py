"""
Tests for Single-File Timer Role

Tests the new LLM-friendly single-file timer role that consolidates
all timer functionality into one file following the new architecture patterns.

Following TDD principles - tests written first.
"""

import time
from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.enhanced_event_context import (
    LLMSafeEventContext,
    create_context_from_event_data,
)
from common.intents import AuditIntent, Intent, NotificationIntent


class TestSingleFileTimerRole:
    """Test single-file timer role functionality."""

    def test_timer_role_config_structure(self):
        """Test that timer role has proper config structure."""
        from roles.timer_single_file import ROLE_CONFIG

        assert "name" in ROLE_CONFIG
        assert ROLE_CONFIG["name"] == "timer"
        assert "version" in ROLE_CONFIG
        assert "description" in ROLE_CONFIG
        assert "llm_type" in ROLE_CONFIG
        assert ROLE_CONFIG["llm_type"] == "WEAK"
        assert "fast_reply" in ROLE_CONFIG
        assert "when_to_use" in ROLE_CONFIG

    def test_timer_intent_definition(self):
        """Test timer-specific intent definition."""
        from roles.timer_single_file import TimerIntent

        # Create valid timer intent
        intent = TimerIntent(action="create", duration=300, label="Test timer")

        assert intent.validate() is True
        assert intent.action == "create"
        assert intent.duration == 300
        assert intent.label == "Test timer"

    def test_timer_intent_validation(self):
        """Test timer intent validation."""
        from roles.timer_single_file import TimerIntent

        # Valid intent
        valid_intent = TimerIntent(action="create", duration=60)
        assert valid_intent.validate() is True

        # Invalid action
        invalid_intent = TimerIntent(action="invalid_action")
        assert invalid_intent.validate() is False

        # Empty action
        empty_intent = TimerIntent(action="")
        assert empty_intent.validate() is False

    def test_timer_expiry_handler(self):
        """Test pure function timer expiry handler."""
        from roles.timer_single_file import handle_timer_expiry

        # Create test context
        context = create_context_from_event_data(
            ["timer_123", "Set timer for 5 minutes"],
            source="timer",
            user_id="user123",
            channel_id="channel456",
        )

        # Call handler
        intents = handle_timer_expiry(["timer_123", "Set timer for 5 minutes"], context)

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

    def test_timer_expiry_handler_error_handling(self):
        """Test timer expiry handler error handling."""
        from roles.timer_single_file import handle_timer_expiry

        # Create problematic context
        context = create_context_from_event_data(None, source="timer")

        # Handler should not raise exception
        intents = handle_timer_expiry(None, context)

        # Should return error handling intents
        assert isinstance(intents, list)
        assert len(intents) >= 1

        # Should contain notification about error
        notification_intents = [i for i in intents if isinstance(i, NotificationIntent)]
        assert len(notification_intents) >= 1

        # Error notification should have high priority
        error_notifications = [
            i
            for i in notification_intents
            if i.priority == "high" or "error" in i.message.lower()
        ]
        assert len(error_notifications) >= 1

    def test_heartbeat_monitoring_handler(self):
        """Test heartbeat monitoring handler."""
        from roles.timer_single_file import handle_heartbeat_monitoring

        context = create_context_from_event_data({}, source="heartbeat")
        intents = handle_heartbeat_monitoring({}, context)

        # Should return intents for timer monitoring
        assert isinstance(intents, list)
        # Heartbeat monitoring might return empty list if no action needed
        assert all(intent.validate() for intent in intents if intents)

    def test_timer_tools(self):
        """Test timer tools return proper data."""
        from roles.timer_single_file import cancel_timer, list_timers, set_timer

        # Test set_timer tool
        result = set_timer("5m", "Test timer")
        assert result["success"] is True
        assert "message" in result
        assert "Timer set" in result["message"]

        # Test cancel_timer tool
        result = cancel_timer("timer_123")
        assert result["success"] is True
        assert "cancelled" in result["message"]

        # Test list_timers tool
        result = list_timers()
        assert result["success"] is True
        assert "message" in result

    def test_timer_helper_functions(self):
        """Test timer helper functions."""
        from roles.timer_single_file import _parse_duration, _parse_timer_event_data

        # Test event data parsing
        timer_id, request = _parse_timer_event_data(["timer_123", "Test request"])
        assert timer_id == "timer_123"
        assert request == "Test request"

        # Test dict parsing
        timer_id, request = _parse_timer_event_data(
            {"timer_id": "timer_456", "original_request": "Dict request"}
        )
        assert timer_id == "timer_456"
        assert request == "Dict request"

        # Test duration parsing
        assert _parse_duration("5m") == 300
        assert _parse_duration("1h") == 3600
        assert _parse_duration("30s") == 30
        assert _parse_duration("120") == 120

    def test_timer_helper_error_handling(self):
        """Test timer helper function error handling."""
        from roles.timer_single_file import _parse_duration, _parse_timer_event_data

        # Test malformed data parsing
        timer_id, request = _parse_timer_event_data("invalid_data")
        assert "parse_error" in timer_id or "unknown" in timer_id

        # Test invalid duration
        with pytest.raises(ValueError):
            _parse_duration("invalid_duration")

    def test_role_registration(self):
        """Test role registration function."""
        from roles.timer_single_file import ROLE_CONFIG, register_role

        role_info = register_role()

        # Should have all required sections
        assert "config" in role_info
        assert "event_handlers" in role_info
        assert "tools" in role_info
        assert "intents" in role_info

        # Config should match ROLE_CONFIG
        assert role_info["config"] == ROLE_CONFIG

        # Should have event handlers
        assert "TIMER_EXPIRED" in role_info["event_handlers"]
        assert "FAST_HEARTBEAT_TICK" in role_info["event_handlers"]

        # Should have tools
        assert len(role_info["tools"]) >= 3  # set_timer, cancel_timer, list_timers

        # Should have intents
        assert len(role_info["intents"]) >= 1  # TimerIntent

    def test_timer_role_file_structure(self):
        """Test that timer role follows single-file structure."""
        import roles.timer_single_file as timer_role

        # Should have all required components
        assert hasattr(timer_role, "ROLE_CONFIG")
        assert hasattr(timer_role, "TimerIntent")
        assert hasattr(timer_role, "handle_timer_expiry")
        assert hasattr(timer_role, "handle_heartbeat_monitoring")
        assert hasattr(timer_role, "set_timer")
        assert hasattr(timer_role, "cancel_timer")
        assert hasattr(timer_role, "list_timers")
        assert hasattr(timer_role, "register_role")

    def test_timer_role_llm_safety(self):
        """Test that timer role follows LLM-safe patterns."""
        from roles.timer_single_file import handle_timer_expiry

        # Handler should be pure function (no side effects)
        context = create_context_from_event_data(["timer_123", "Test"], source="timer")

        # Multiple calls should return consistent results
        result1 = handle_timer_expiry(["timer_123", "Test"], context)
        result2 = handle_timer_expiry(["timer_123", "Test"], context)

        # Results should be consistent (same number of intents)
        assert len(result1) == len(result2)
        assert all(isinstance(i, Intent) for i in result1)
        assert all(isinstance(i, Intent) for i in result2)


class TestTimerRoleIntegration:
    """Test timer role integration with new architecture."""

    def test_timer_role_with_intent_processor(self):
        """Test timer role integration with intent processor."""
        from common.intent_processor import IntentProcessor
        from roles.timer_single_file import TimerIntent, handle_timer_expiry

        # Create mock processor
        processor = IntentProcessor()

        # Create context and call handler
        context = create_context_from_event_data(
            ["timer_123", "Test timer"], source="timer", user_id="user123"
        )

        intents = handle_timer_expiry(["timer_123", "Test timer"], context)

        # Should be processable by intent processor
        assert all(intent.validate() for intent in intents)

    def test_timer_role_with_enhanced_message_bus(self):
        """Test timer role integration with enhanced MessageBus."""
        from common.message_bus import MessageBus
        from roles.timer_single_file import handle_timer_expiry

        # Create enhanced message bus
        bus = MessageBus()
        bus.start()

        # Subscribe timer handler
        bus.subscribe("timer", "TIMER_EXPIRED", handle_timer_expiry)

        # Should be registered successfully
        assert "TIMER_EXPIRED" in bus._subscribers
        assert "timer" in bus._subscribers["TIMER_EXPIRED"]

    def test_timer_role_migration_compatibility(self):
        """Test that single-file timer role maintains compatibility."""
        from roles.timer_single_file import register_role

        role_info = register_role()

        # Should have same interface as multi-file roles
        assert "config" in role_info
        assert "event_handlers" in role_info
        assert "tools" in role_info

        # Config should have expected timer role fields
        config = role_info["config"]
        assert config["name"] == "timer"
        assert config["llm_type"] == "WEAK"
        assert config["fast_reply"] is True


if __name__ == "__main__":
    pytest.main([__file__])
