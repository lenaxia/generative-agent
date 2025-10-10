"""Unit tests for Timer Tools functionality.

Tests the timer tools that provide the interface between the role system
and the TimerManager for actual timer operations.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roles.timer.tools import (
    alarm_cancel,
    alarm_set,
    timer_cancel,
    timer_list,
    timer_set,
    timer_snooze,
)


class TestTimerTools:
    """Test suite for timer tools."""

    @pytest.fixture
    def mock_timer_manager(self):
        """Mock TimerManager for testing."""
        manager_mock = AsyncMock()
        manager_mock.create_timer = AsyncMock()
        manager_mock.cancel_timer = AsyncMock()
        manager_mock.list_timers = AsyncMock()
        manager_mock.snooze_timer = AsyncMock()
        return manager_mock

    @pytest.fixture
    def sample_timer_data(self):
        """Sample timer data for testing."""
        return {
            "id": "timer_123",
            "name": "Test Timer",
            "type": "countdown",
            "status": "active",
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + 3600,
            "duration_seconds": 3600,
            "user_id": "user123",
            "channel_id": "slack:general",
            "label": "Test Label",
        }

    def test_timer_set_success(self, mock_timer_manager, sample_timer_data):
        """Test successful timer creation."""
        # Setup
        mock_timer_manager.create_timer.return_value = "timer_123"

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_set(
                duration="30m",
                label="Test Timer",
                custom_message="Timer expired!",
                user_id="user123",
                channel_id="slack:general",
            )

        # Verify
        assert result["success"] is True
        assert result["timer_id"] == "timer_123"
        assert result["duration"] == "30m"
        assert result["duration_seconds"] == 1800
        assert "Timer set for 30m" in result["message"]

    def test_timer_set_invalid_duration(self, mock_timer_manager):
        """Test timer creation with invalid duration."""
        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_set(duration="invalid_duration")

        # Verify
        assert result["success"] is False
        assert "Invalid duration format" in result["error"]

    def test_timer_set_exception(self, mock_timer_manager):
        """Test timer creation with exception."""
        # Setup
        mock_timer_manager.create_timer.side_effect = Exception("Redis error")

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_set(duration="30m")

        # Verify
        assert result["success"] is False
        assert "Redis error" in result["error"]

    def test_timer_cancel_success(self, mock_timer_manager):
        """Test successful timer cancellation."""
        # Setup
        mock_timer_manager.cancel_timer.return_value = True

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_cancel(timer_id="timer_123")

        # Verify
        assert result["success"] is True
        assert result["timer_id"] == "timer_123"
        assert "cancelled" in result["message"]

    def test_timer_cancel_not_found(self, mock_timer_manager):
        """Test timer cancellation when timer not found."""
        # Setup
        mock_timer_manager.cancel_timer.return_value = False

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_cancel(timer_id="nonexistent")

        # Verify
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_timer_list_success(self, mock_timer_manager, sample_timer_data):
        """Test successful timer listing."""
        # Setup
        mock_timer_manager.list_timers.return_value = [sample_timer_data]

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_list(active_only=True, user_id="user123")

        # Verify
        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["timers"]) == 1

        timer = result["timers"][0]
        assert timer["timer_id"] == "timer_123"
        assert timer["name"] == "Test Timer"
        assert timer["status"] == "active"

    def test_timer_list_empty(self, mock_timer_manager):
        """Test timer listing with no timers."""
        # Setup
        mock_timer_manager.list_timers.return_value = []

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_list()

        # Verify
        assert result["success"] is True
        assert result["count"] == 0
        assert len(result["timers"]) == 0

    def test_timer_list_exception(self, mock_timer_manager):
        """Test timer listing with exception."""
        # Setup
        mock_timer_manager.list_timers.side_effect = Exception("Database error")

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_list()

        # Verify
        assert result["success"] is False
        assert "Database error" in result["error"]
        assert result["timers"] == []

    def test_alarm_set_success(self, mock_timer_manager):
        """Test successful alarm creation."""
        # Setup
        mock_timer_manager.create_timer.return_value = "alarm_123"

        # Use a future time for the alarm (ensure it's actually in the future)
        future_datetime = datetime.now() + timedelta(hours=1)
        # If we cross midnight, use a time later today instead
        if future_datetime.date() > datetime.now().date():
            future_datetime = datetime.now().replace(hour=23, minute=59)
        future_time = future_datetime.strftime("%H:%M")

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = alarm_set(
                time=future_time,
                label="Morning Alarm",
                recurring="daily",
                user_id="user123",
            )

        # Verify
        assert result["success"] is True
        assert result["timer_id"] == "alarm_123"
        assert result["time"] == future_time
        assert result["recurring"] == "daily"

    def test_alarm_set_invalid_time(self, mock_timer_manager):
        """Test alarm creation with invalid time."""
        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = alarm_set(time="invalid_time")

        # Verify
        assert result["success"] is False
        assert "Invalid time format" in result["error"]

    def test_alarm_set_past_time(self, mock_timer_manager):
        """Test alarm creation with past time."""
        # Use a past time (more than 2 hours ago to avoid auto-tomorrow logic)
        past_time = (datetime.now() - timedelta(hours=3)).strftime("%H:%M")

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = alarm_set(time=past_time)

        # Verify
        assert result["success"] is False
        assert "must be in the future" in result["error"]

    def test_alarm_cancel_success(self, mock_timer_manager):
        """Test successful alarm cancellation."""
        # Setup
        mock_timer_manager.cancel_timer.return_value = True

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = alarm_cancel(alarm_id="alarm_123")

        # Verify
        assert result["success"] is True
        assert result["timer_id"] == "alarm_123"

    def test_timer_snooze_success(self, mock_timer_manager):
        """Test successful timer snoozing."""
        # Setup
        mock_timer_manager.snooze_timer.return_value = True

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_snooze(timer_id="timer_123", snooze_minutes=10)

        # Verify
        assert result["success"] is True
        assert result["timer_id"] == "timer_123"
        assert result["snooze_minutes"] == 10
        assert "snoozed for 10 minutes" in result["message"]

    def test_timer_snooze_invalid_minutes(self, mock_timer_manager):
        """Test timer snoozing with invalid minutes."""
        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_snooze(timer_id="timer_123", snooze_minutes=-5)

        # Verify
        assert result["success"] is False
        assert "must be positive" in result["error"]

    def test_timer_snooze_not_found(self, mock_timer_manager):
        """Test timer snoozing when timer not found."""
        # Setup
        mock_timer_manager.snooze_timer.return_value = False

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Execute
            result = timer_snooze(timer_id="nonexistent")

        # Verify
        assert result["success"] is False
        assert "not found or not active" in result["error"]

    def test_duration_parsing_edge_cases(self, mock_timer_manager):
        """Test various duration formats."""
        test_cases = [
            ("5m", 300),
            ("1h", 3600),
            ("1h30m", 5400),
            ("90", 90),
            ("2h15m30s", 8130),
        ]

        mock_timer_manager.create_timer.return_value = "timer_test"

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            for duration_str, expected_seconds in test_cases:
                result = timer_set(duration=duration_str)

                assert result["success"] is True
                assert result["duration_seconds"] == expected_seconds

    def test_timer_list_filtering(self, mock_timer_manager, sample_timer_data):
        """Test timer listing with different filters."""
        # Setup
        mock_timer_manager.list_timers.return_value = [sample_timer_data]

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Test user filtering
            result = timer_list(user_id="user123")
            assert result["success"] is True

            # Test channel filtering
            result = timer_list(channel_id="slack:general")
            assert result["success"] is True

            # Test active only
            result = timer_list(active_only=True)
            assert result["success"] is True

            # Test all timers
            result = timer_list(active_only=False)
            assert result["success"] is True

    def test_alarm_recurring_patterns(self, mock_timer_manager):
        """Test different recurring patterns for alarms."""
        recurring_patterns = ["daily", "weekly", "weekdays", "weekends"]
        future_time = (datetime.now() + timedelta(hours=1)).strftime("%H:%M")

        mock_timer_manager.create_timer.return_value = "alarm_recurring"

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            for pattern in recurring_patterns:
                result = alarm_set(time=future_time, recurring=pattern)

                assert result["success"] is True
                assert result["recurring"] == pattern

    def test_timer_tools_integration(self, mock_timer_manager):
        """Test integration between different timer tools."""
        # Setup
        mock_timer_manager.create_timer.return_value = "timer_integration"
        mock_timer_manager.list_timers.return_value = []
        mock_timer_manager.snooze_timer.return_value = True
        mock_timer_manager.cancel_timer.return_value = True

        with patch(
            "roles.timer.tools.get_timer_manager", return_value=mock_timer_manager
        ):
            # Create timer
            create_result = timer_set(duration="15m", label="Integration Test")
            assert create_result["success"] is True
            timer_id = create_result["timer_id"]

            # List timers
            list_result = timer_list()
            assert list_result["success"] is True

            # Snooze timer
            snooze_result = timer_snooze(timer_id=timer_id, snooze_minutes=5)
            assert snooze_result["success"] is True

            # Cancel timer
            cancel_result = timer_cancel(timer_id=timer_id)
            assert cancel_result["success"] is True
