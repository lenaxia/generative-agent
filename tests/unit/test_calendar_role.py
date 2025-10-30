"""
Tests for calendar role implementation.

This module tests the calendar role that provides context-aware scheduling
and calendar management functionality.
"""

import time
from unittest.mock import Mock

import pytest

from common.event_context import LLMSafeEventContext
from common.intents import Intent
from roles.core_calendar import (
    ROLE_CONFIG,
    CalendarIntent,
    add_calendar_event,
    get_schedule,
    handle_calendar_request,
    register_role,
)


class TestCalendarRoleConfig:
    """Test calendar role configuration."""

    def test_role_config_structure(self):
        """Test calendar role configuration structure."""
        assert ROLE_CONFIG["name"] == "calendar"
        assert ROLE_CONFIG["version"] == "1.0.0"
        assert "Calendar and scheduling management" in ROLE_CONFIG["description"]
        assert ROLE_CONFIG["llm_type"] == "DEFAULT"
        assert ROLE_CONFIG["fast_reply"] is True
        assert ROLE_CONFIG["memory_enabled"] is True
        assert ROLE_CONFIG["location_aware"] is True

    def test_role_config_when_to_use(self):
        """Test when_to_use guidance."""
        when_to_use = ROLE_CONFIG["when_to_use"]

        assert "schedule management" in when_to_use.lower()
        assert "calendar queries" in when_to_use.lower()
        assert "event planning" in when_to_use.lower()


class TestCalendarIntent:
    """Test CalendarIntent dataclass."""

    def test_calendar_intent_creation(self):
        """Test CalendarIntent creation."""
        intent = CalendarIntent(
            action="add_event",
            event_data={
                "title": "Meeting with Bob",
                "start_time": "2023-10-15T14:00:00",
                "duration": 60,
                "location": "office",
            },
        )

        assert intent.action == "add_event"
        assert intent.event_data["title"] == "Meeting with Bob"
        assert intent.event_data["start_time"] == "2023-10-15T14:00:00"
        assert intent.event_data["duration"] == 60
        assert intent.event_data["location"] == "office"

    def test_calendar_intent_validation_valid_actions(self):
        """Test CalendarIntent validation with valid actions."""
        valid_actions = ["add_event", "get_schedule", "find_conflicts"]

        for action in valid_actions:
            intent = CalendarIntent(action=action, event_data={"test": "data"})
            assert intent.validate() is True

    def test_calendar_intent_validation_invalid_action(self):
        """Test CalendarIntent validation with invalid action."""
        intent = CalendarIntent(action="invalid_action", event_data={"test": "data"})

        assert intent.validate() is False

    def test_calendar_intent_validation_empty_action(self):
        """Test CalendarIntent validation with empty action."""
        intent = CalendarIntent(action="", event_data={"test": "data"})

        assert intent.validate() is False


class TestCalendarEventHandlers:
    """Test calendar event handlers."""

    @pytest.fixture
    def mock_context(self):
        """Create mock event context."""
        context = Mock(spec=LLMSafeEventContext)
        context.user_id = "test_user"
        context.channel_id = "console"
        context.to_dict.return_value = {
            "user_id": "test_user",
            "channel_id": "console",
            "timestamp": time.time(),
        }
        return context

    def test_handle_calendar_request_basic(self, mock_context):
        """Test basic calendar request handling."""
        event_data = {"request": "What's my schedule for today?"}

        intents = handle_calendar_request(event_data, mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], CalendarIntent)
        assert intents[0].action == "get_schedule"
        assert intents[0].event_data["query"] == "What's my schedule for today?"
        assert intents[0].event_data["context"] == mock_context.to_dict()

    def test_handle_calendar_request_empty_request(self, mock_context):
        """Test calendar request handling with empty request."""
        event_data = {"request": ""}

        intents = handle_calendar_request(event_data, mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], CalendarIntent)
        assert intents[0].event_data["query"] == ""

    def test_handle_calendar_request_missing_request(self, mock_context):
        """Test calendar request handling with missing request field."""
        event_data = {}

        intents = handle_calendar_request(event_data, mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], CalendarIntent)
        assert intents[0].event_data["query"] == ""


class TestCalendarTools:
    """Test calendar tool functions."""

    def test_get_schedule_basic(self):
        """Test basic schedule retrieval."""
        result = get_schedule(user_id="test_user", days_ahead=7)

        assert result["success"] is True
        assert "events" in result
        assert isinstance(result["events"], list)
        assert "Schedule retrieved for test_user" in result["message"]

    def test_get_schedule_with_location(self):
        """Test schedule retrieval with location context."""
        result = get_schedule(user_id="test_user", days_ahead=3, location="office")

        assert result["success"] is True
        assert "location: office" in result["message"]

    def test_get_schedule_custom_days(self):
        """Test schedule retrieval with custom days ahead."""
        result = get_schedule(user_id="test_user", days_ahead=14)

        assert result["success"] is True
        assert "Schedule retrieved for test_user" in result["message"]

    def test_add_calendar_event_basic(self):
        """Test basic calendar event addition."""
        result = add_calendar_event(
            title="Team Meeting",
            start_time="2023-10-15T10:00:00",
            duration=60,
            user_id="test_user",
        )

        assert result["success"] is True
        assert "event_id" in result
        assert result["event_id"].startswith("evt_")
        assert "Added event: Team Meeting at 2023-10-15T10:00:00" in result["message"]

    def test_add_calendar_event_with_location(self):
        """Test calendar event addition with location."""
        result = add_calendar_event(
            title="Client Meeting",
            start_time="2023-10-15T14:00:00",
            duration=90,
            location="conference_room",
            user_id="test_user",
        )

        assert result["success"] is True
        assert "event_id" in result
        assert "Added event: Client Meeting at 2023-10-15T14:00:00" in result["message"]

    def test_add_calendar_event_default_duration(self):
        """Test calendar event addition with default duration."""
        result = add_calendar_event(
            title="Quick Call", start_time="2023-10-15T16:00:00", user_id="test_user"
        )

        assert result["success"] is True
        # Default duration should be 60 minutes
        assert "Added event: Quick Call at 2023-10-15T16:00:00" in result["message"]


class TestCalendarRoleRegistration:
    """Test calendar role registration."""

    def test_register_role_structure(self):
        """Test role registration structure."""
        registration = register_role()

        assert "config" in registration
        assert "event_handlers" in registration
        assert "tools" in registration
        assert "intents" in registration

    def test_register_role_config(self):
        """Test registered role config."""
        registration = register_role()

        assert registration["config"] == ROLE_CONFIG

    def test_register_role_event_handlers(self):
        """Test registered event handlers."""
        registration = register_role()

        assert "CALENDAR_REQUEST" in registration["event_handlers"]
        assert (
            registration["event_handlers"]["CALENDAR_REQUEST"]
            == handle_calendar_request
        )

    def test_register_role_tools(self):
        """Test registered tools."""
        registration = register_role()

        tools = registration["tools"]
        assert len(tools) == 2
        assert get_schedule in tools
        assert add_calendar_event in tools

    def test_register_role_intents(self):
        """Test registered intents."""
        registration = register_role()

        intents = registration["intents"]
        assert len(intents) == 1
        assert CalendarIntent in intents


class TestCalendarRoleIntegration:
    """Test calendar role integration scenarios."""

    def test_calendar_role_context_awareness(self):
        """Test that calendar role is designed for context awareness."""
        # Verify role config indicates context awareness
        assert ROLE_CONFIG["memory_enabled"] is True
        assert ROLE_CONFIG["location_aware"] is True

    def test_calendar_intent_inheritance(self):
        """Test that CalendarIntent properly inherits from Intent."""
        intent = CalendarIntent(
            action="get_schedule", event_data={"query": "today's events"}
        )

        assert isinstance(intent, Intent)
        assert hasattr(intent, "validate")
        assert hasattr(intent, "to_dict")

    def test_calendar_tools_return_format(self):
        """Test that calendar tools return consistent format."""
        # Test get_schedule return format
        schedule_result = get_schedule("test_user")
        assert "success" in schedule_result
        assert "events" in schedule_result
        assert "message" in schedule_result

        # Test add_calendar_event return format
        event_result = add_calendar_event("Test Event", "2023-10-15T10:00:00")
        assert "success" in event_result
        assert "event_id" in event_result
        assert "message" in event_result


if __name__ == "__main__":
    pytest.main([__file__])
