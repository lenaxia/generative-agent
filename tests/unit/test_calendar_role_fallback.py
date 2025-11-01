"""Tests for calendar role graceful fallback when not configured."""

import os

# Import the tools directly
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCalendarRoleFallback:
    """Test calendar role graceful fallback behavior."""

    @patch.dict(os.environ, {}, clear=True)
    def test_get_schedule_not_configured(self):
        """Test get_schedule returns graceful error when not configured."""
        from roles.core_calendar import get_schedule

        result = get_schedule(user_id="test_user", days_ahead=7)

        assert result["success"] is False
        assert result["events"] == []
        assert "not configured" in result["message"].lower()
        assert "CALDAV_URL" in result["message"]

    @patch.dict(os.environ, {}, clear=True)
    def test_add_calendar_event_not_configured(self):
        """Test add_calendar_event returns graceful error when not configured."""
        from roles.core_calendar import add_calendar_event

        result = add_calendar_event(
            title="Test Event", start_time="2024-03-15T14:00:00", duration=60
        )

        assert result["success"] is False
        assert result["event_id"] is None
        assert "not configured" in result["message"].lower()
        assert "CALDAV_URL" in result["message"]

    @patch.dict(
        os.environ,
        {
            "CALDAV_URL": "https://caldav.example.com",
            "CALDAV_USERNAME": "",  # Missing username
            "CALDAV_PASSWORD": "password",
        },
    )
    def test_get_schedule_partially_configured(self):
        """Test get_schedule handles partial configuration gracefully."""
        from roles.core_calendar import get_schedule

        result = get_schedule(user_id="test_user", days_ahead=7)

        assert result["success"] is False
        assert result["events"] == []
        assert "not configured" in result["message"].lower()

    @patch.dict(
        os.environ,
        {
            "CALDAV_URL": "https://caldav.example.com",
            "CALDAV_USERNAME": "user@example.com",
            "CALDAV_PASSWORD": "password",
        },
    )
    @patch("roles.core_calendar.get_calendar_provider")
    def test_get_schedule_provider_error(self, mock_get_provider):
        """Test get_schedule handles provider errors gracefully."""
        from roles.core_calendar import get_schedule

        # Simulate provider error
        mock_get_provider.side_effect = Exception("Connection failed")

        result = get_schedule(user_id="test_user", days_ahead=7)

        assert result["success"] is False
        assert result["events"] == []
        assert "error" in result["message"].lower()
        assert "configuration" in result["message"].lower()

    @patch.dict(
        os.environ,
        {
            "CALDAV_URL": "https://caldav.example.com",
            "CALDAV_USERNAME": "user@example.com",
            "CALDAV_PASSWORD": "password",
        },
    )
    @patch("roles.core_calendar.get_calendar_provider")
    def test_add_event_provider_error(self, mock_get_provider):
        """Test add_calendar_event handles provider errors gracefully."""
        from roles.core_calendar import add_calendar_event

        # Simulate provider error
        mock_get_provider.side_effect = Exception("Authentication failed")

        result = add_calendar_event(
            title="Test Event", start_time="2024-03-15T14:00:00", duration=60
        )

        assert result["success"] is False
        assert result["event_id"] is None
        assert "error" in result["message"].lower()
        assert "configuration" in result["message"].lower()

    @patch.dict(
        os.environ,
        {
            "CALDAV_URL": "https://caldav.example.com",
            "CALDAV_USERNAME": "user@example.com",
            "CALDAV_PASSWORD": "password",
        },
    )
    @patch("roles.core_calendar.get_calendar_provider")
    def test_get_schedule_success(self, mock_get_provider):
        """Test get_schedule works when properly configured."""
        from datetime import datetime

        from roles.core_calendar import get_schedule

        # Setup mock provider
        mock_provider = Mock()
        mock_provider.get_events.return_value = [
            {
                "id": "event-1",
                "title": "Test Event",
                "start": datetime(2024, 3, 15, 14, 0),
                "end": datetime(2024, 3, 15, 15, 0),
                "location": "Office",
                "description": "Test",
            }
        ]
        mock_get_provider.return_value = mock_provider

        result = get_schedule(user_id="test_user", days_ahead=7)

        assert result["success"] is True
        assert len(result["events"]) == 1
        assert result["events"][0]["title"] == "Test Event"
        assert "Retrieved 1 events" in result["message"]

    @patch.dict(
        os.environ,
        {
            "CALDAV_URL": "https://caldav.example.com",
            "CALDAV_USERNAME": "user@example.com",
            "CALDAV_PASSWORD": "password",
        },
    )
    @patch("roles.core_calendar.get_calendar_provider")
    def test_add_event_success(self, mock_get_provider):
        """Test add_calendar_event works when properly configured."""
        from datetime import datetime

        from roles.core_calendar import add_calendar_event

        # Setup mock provider
        mock_provider = Mock()
        mock_provider.add_event.return_value = {
            "id": "event-123",
            "title": "Test Event",
            "start": datetime(2024, 3, 15, 14, 0),
            "end": datetime(2024, 3, 15, 15, 0),
            "location": "Office",
            "description": "Test",
        }
        mock_get_provider.return_value = mock_provider

        result = add_calendar_event(
            title="Test Event",
            start_time="2024-03-15T14:00:00",
            duration=60,
            location="Office",
        )

        assert result["success"] is True
        assert result["event_id"] == "event-123"
        assert "Added event: Test Event" in result["message"]
