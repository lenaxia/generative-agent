"""Tests for realtime log integration with roles."""

from unittest.mock import MagicMock, patch

import pytest


class TestConversationRealtimeIntegration:
    """Test conversation role writes to realtime log."""

    def test_conversation_writes_to_realtime_log(self):
        """Test conversation post-processing writes to realtime log."""
        with patch("common.realtime_log.add_message") as mock_add:
            mock_add.return_value = True

            from roles.core_conversation import save_message_to_log

            context = MagicMock()
            context.user_id = "test_user"
            context.original_prompt = "Test user message"

            result = save_message_to_log("Test assistant response", context, {})

            assert result == "Test assistant response"
            mock_add.assert_called_once_with(
                user_id="test_user",
                user_message="Test user message",
                assistant_response="Test assistant response",
                role="conversation",
                metadata=None,
            )

    def test_conversation_handles_realtime_log_failure(self):
        """Test conversation handles realtime log write failure gracefully."""
        with patch("common.realtime_log.add_message") as mock_add:
            mock_add.return_value = False

            from roles.core_conversation import save_message_to_log

            context = MagicMock()
            context.user_id = "test_user"
            context.original_prompt = "Test"

            # Should still return result even if realtime log fails
            result = save_message_to_log("Response", context, {})
            assert result == "Response"


class TestCalendarRealtimeIntegration:
    """Test calendar role writes to realtime log."""

    def test_calendar_writes_to_realtime_log(self):
        """Test calendar post-processing writes to realtime log."""
        with patch("common.realtime_log.add_message") as mock_add:
            mock_add.return_value = True

            from roles.core_calendar import save_calendar_event

            context = MagicMock()
            context.user_id = "test_user"
            context.original_prompt = "Schedule a meeting"

            result = save_calendar_event("Meeting scheduled for tomorrow", context, {})

            assert result == "Meeting scheduled for tomorrow"
            mock_add.assert_called_once_with(
                user_id="test_user",
                user_message="Schedule a meeting",
                assistant_response="Meeting scheduled for tomorrow",
                role="calendar",
                metadata=None,
            )

    def test_calendar_handles_realtime_log_failure(self):
        """Test calendar handles realtime log write failure gracefully."""
        with patch("common.realtime_log.add_message") as mock_add:
            mock_add.return_value = False

            from roles.core_calendar import save_calendar_event

            context = MagicMock()
            context.user_id = "test_user"
            context.original_prompt = "Test"

            result = save_calendar_event("Response", context, {})
            assert result == "Response"


class TestPlanningRealtimeIntegration:
    """Test planning role writes to realtime log."""

    def test_planning_writes_to_realtime_log(self):
        """Test planning post-processing writes to realtime log."""
        with patch("common.realtime_log.add_message") as mock_add:
            mock_add.return_value = True

            from roles.core_planning import save_planning_result

            context = MagicMock()
            context.user_id = "test_user"
            context.original_prompt = "Plan a trip"

            result = save_planning_result("Created 3-task plan", context, {})

            assert result == "Created 3-task plan"
            mock_add.assert_called_once_with(
                user_id="test_user",
                user_message="Plan a trip",
                assistant_response="Created 3-task plan",
                role="planning",
                metadata=None,
            )

    def test_planning_handles_realtime_log_failure(self):
        """Test planning handles realtime log write failure gracefully."""
        with patch("common.realtime_log.add_message") as mock_add:
            mock_add.return_value = False

            from roles.core_planning import save_planning_result

            context = MagicMock()
            context.user_id = "test_user"
            context.original_prompt = "Test"

            result = save_planning_result("Response", context, {})
            assert result == "Response"
