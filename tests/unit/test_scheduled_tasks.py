"""Tests for scheduled tasks including conversation inactivity checking.

This module tests the scheduled tasks that run periodically to handle
background operations like conversation inactivity checking.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supervisor.scheduled_tasks import check_conversation_inactivity


@pytest.fixture
def mock_users_with_unanalyzed():
    """Create mock users with unanalyzed messages."""
    return ["user1", "user2", "user3"]


@pytest.mark.asyncio
async def test_inactivity_checker_triggers_analysis():
    """Test analysis triggered after timeout."""
    with (
        patch(
            "supervisor.scheduled_tasks.get_unanalyzed_messages"
        ) as mock_get_unanalyzed,
        patch("supervisor.scheduled_tasks.get_last_message_time") as mock_get_last_time,
        patch("supervisor.scheduled_tasks.analyze_conversation") as mock_analyze,
    ):
        # Setup: user1 has unanalyzed messages and last message > 30 min ago
        mock_get_unanalyzed.side_effect = lambda user_id: (
            [{"id": "msg1"}] if user_id == "user1" else []
        )
        current_time = time.time()
        mock_get_last_time.return_value = current_time - (31 * 60)  # 31 minutes ago
        mock_analyze.return_value = AsyncMock(
            return_value={
                "success": True,
                "analyzed_count": 1,
                "memories_created": 1,
            }
        )()

        # Execute
        result = await check_conversation_inactivity(
            user_ids=["user1"], inactivity_timeout_minutes=30
        )

        # Verify
        assert result["success"] is True
        assert result["users_analyzed"] == 1
        mock_analyze.assert_called_once_with(user_id="user1")


@pytest.mark.asyncio
async def test_inactivity_checker_no_trigger_recent():
    """Test no trigger for recent activity."""
    with (
        patch(
            "supervisor.scheduled_tasks.get_unanalyzed_messages"
        ) as mock_get_unanalyzed,
        patch("supervisor.scheduled_tasks.get_last_message_time") as mock_get_last_time,
        patch("supervisor.scheduled_tasks.analyze_conversation") as mock_analyze,
    ):
        # Setup: user has unanalyzed messages but last message < 30 min ago
        mock_get_unanalyzed.return_value = [{"id": "msg1"}]
        current_time = time.time()
        mock_get_last_time.return_value = current_time - (10 * 60)  # 10 minutes ago
        mock_analyze.return_value = {
            "success": True,
            "analyzed_count": 0,
            "memories_created": 0,
        }

        # Execute
        result = await check_conversation_inactivity(
            user_ids=["user1"], inactivity_timeout_minutes=30
        )

        # Verify - should not trigger analysis
        assert result["success"] is True
        assert result["users_analyzed"] == 0
        mock_analyze.assert_not_called()


@pytest.mark.asyncio
async def test_inactivity_checker_multiple_users():
    """Test handles multiple users."""
    with (
        patch(
            "supervisor.scheduled_tasks.get_unanalyzed_messages"
        ) as mock_get_unanalyzed,
        patch("supervisor.scheduled_tasks.get_last_message_time") as mock_get_last_time,
        patch("supervisor.scheduled_tasks.analyze_conversation") as mock_analyze,
    ):
        # Setup: user1 and user3 need analysis, user2 doesn't
        mock_get_unanalyzed.side_effect = lambda user_id: (
            [{"id": f"msg_{user_id}"}] if user_id in ["user1", "user3"] else []
        )
        current_time = time.time()

        def get_time(user_id):
            if user_id == "user1":
                return current_time - (35 * 60)  # 35 minutes ago
            elif user_id == "user2":
                return current_time - (5 * 60)  # 5 minutes ago
            elif user_id == "user3":
                return current_time - (40 * 60)  # 40 minutes ago
            return None

        mock_get_last_time.side_effect = get_time

        async def mock_analyze_func(user_id):
            return {
                "success": True,
                "analyzed_count": 1,
                "memories_created": 1,
            }

        mock_analyze.side_effect = mock_analyze_func

        # Execute
        result = await check_conversation_inactivity(
            user_ids=["user1", "user2", "user3"], inactivity_timeout_minutes=30
        )

        # Verify - should analyze user1 and user3
        assert result["success"] is True
        assert result["users_analyzed"] == 2
        assert mock_analyze.call_count == 2


@pytest.mark.asyncio
async def test_inactivity_checker_no_unanalyzed():
    """Test skips users with no unanalyzed messages."""
    with (
        patch(
            "supervisor.scheduled_tasks.get_unanalyzed_messages"
        ) as mock_get_unanalyzed,
        patch("supervisor.scheduled_tasks.get_last_message_time") as mock_get_last_time,
        patch("supervisor.scheduled_tasks.analyze_conversation") as mock_analyze,
    ):
        # Setup: no unanalyzed messages
        mock_get_unanalyzed.return_value = []
        current_time = time.time()
        mock_get_last_time.return_value = current_time - (35 * 60)

        # Execute
        result = await check_conversation_inactivity(
            user_ids=["user1"], inactivity_timeout_minutes=30
        )

        # Verify - should not analyze
        assert result["success"] is True
        assert result["users_analyzed"] == 0
        mock_analyze.assert_not_called()


@pytest.mark.asyncio
async def test_inactivity_checker_timeout_configuration():
    """Test respects timeout configuration."""
    with (
        patch(
            "supervisor.scheduled_tasks.get_unanalyzed_messages"
        ) as mock_get_unanalyzed,
        patch("supervisor.scheduled_tasks.get_last_message_time") as mock_get_last_time,
        patch("supervisor.scheduled_tasks.analyze_conversation") as mock_analyze,
    ):
        # Setup: user has unanalyzed messages
        mock_get_unanalyzed.return_value = [{"id": "msg1"}]
        current_time = time.time()
        mock_get_last_time.return_value = current_time - (25 * 60)  # 25 minutes ago
        mock_analyze.return_value = AsyncMock(
            return_value={
                "success": True,
                "analyzed_count": 1,
                "memories_created": 1,
            }
        )()

        # Execute with 20 minute timeout (should trigger)
        result = await check_conversation_inactivity(
            user_ids=["user1"], inactivity_timeout_minutes=20
        )

        # Verify - should trigger with 20 min timeout
        assert result["success"] is True
        assert result["users_analyzed"] == 1
        mock_analyze.assert_called_once()

        # Reset and test with 30 minute timeout (should not trigger)
        mock_analyze.reset_mock()
        result = await check_conversation_inactivity(
            user_ids=["user1"], inactivity_timeout_minutes=30
        )

        # Verify - should not trigger with 30 min timeout
        assert result["success"] is True
        assert result["users_analyzed"] == 0
        mock_analyze.assert_not_called()


@pytest.mark.asyncio
async def test_inactivity_checker_error_handling():
    """Test handles errors gracefully."""
    with (
        patch(
            "supervisor.scheduled_tasks.get_unanalyzed_messages"
        ) as mock_get_unanalyzed,
        patch("supervisor.scheduled_tasks.get_last_message_time") as mock_get_last_time,
        patch("supervisor.scheduled_tasks.analyze_conversation") as mock_analyze,
    ):
        # Setup: first user succeeds, second fails
        mock_get_unanalyzed.return_value = [{"id": "msg1"}]
        current_time = time.time()
        mock_get_last_time.return_value = current_time - (35 * 60)

        # First call succeeds, second raises exception
        async def side_effect_func(user_id):
            if user_id == "user1":
                return {"success": True, "analyzed_count": 1, "memories_created": 1}
            else:
                raise Exception("Analysis failed")

        mock_analyze.side_effect = side_effect_func

        # Execute
        result = await check_conversation_inactivity(
            user_ids=["user1", "user2"], inactivity_timeout_minutes=30
        )

        # Verify - should handle error and continue
        assert result["success"] is True
        assert result["users_analyzed"] == 1
        assert result["errors"] == 1
        assert mock_analyze.call_count == 2
