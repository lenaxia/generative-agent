"""Unit tests for Timer Monitor functionality.

Tests the timer monitoring system that integrates with the Heartbeat
to process expired timers and handle timer events.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.message_bus import MessageBus, MessageType
from supervisor.timer_monitor import TimerMonitor


class TestTimerMonitor:
    """Test suite for TimerMonitor class."""

    @pytest.fixture
    def mock_message_bus(self):
        """Mock MessageBus for testing."""
        bus_mock = MagicMock(spec=MessageBus)
        bus_mock.publish = MagicMock()
        return bus_mock

    @pytest.fixture
    def mock_timer_manager(self):
        """Mock TimerManager for testing."""
        manager_mock = AsyncMock()
        manager_mock.get_expiring_timers = AsyncMock()
        manager_mock.update_timer_status = AsyncMock()
        manager_mock.create_recurring_instance = AsyncMock()
        return manager_mock

    @pytest.fixture
    def timer_monitor(self, mock_message_bus, mock_timer_manager):
        """Create TimerMonitor instance with mocked dependencies."""
        with patch(
            "supervisor.timer_monitor.get_timer_manager",
            return_value=mock_timer_manager,
        ):
            monitor = TimerMonitor(mock_message_bus)
            return monitor

    @pytest.fixture
    def sample_timer(self):
        """Sample timer data for testing."""
        return {
            "id": "timer_123",
            "name": "Test Timer",
            "type": "countdown",
            "status": "active",
            "created_at": int(time.time()) - 3600,
            "expires_at": int(time.time()) - 60,  # Expired 1 minute ago
            "duration_seconds": 3600,
            "user_id": "user123",
            "channel_id": "slack:general",
            "custom_message": "Timer expired!",
            "notification_config": {
                "channels": [
                    {
                        "channel_id": "slack:general",
                        "message": "Timer expired!",
                        "priority": "high",
                    }
                ]
            },
            "actions": [{"type": "notify", "config": {"sound": True}}],
        }

    @pytest.mark.asyncio
    async def test_check_expired_timers_success(
        self, timer_monitor, mock_timer_manager, sample_timer
    ):
        """Test successful expired timer checking."""
        # Setup
        mock_timer_manager.get_expiring_timers.return_value = [sample_timer]

        # Execute
        expired_timers = await timer_monitor.check_expired_timers()

        # Verify
        assert len(expired_timers) == 1
        assert expired_timers[0]["id"] == "timer_123"
        mock_timer_manager.get_expiring_timers.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_expired_timers_rate_limiting(
        self, timer_monitor, mock_timer_manager
    ):
        """Test rate limiting in expired timer checking."""
        # Setup - set last_check to recent time
        timer_monitor.last_check = int(time.time())

        # Execute
        expired_timers = await timer_monitor.check_expired_timers()

        # Verify - should return empty due to rate limiting
        assert len(expired_timers) == 0
        mock_timer_manager.get_expiring_timers.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_expired_timers_exception(
        self, timer_monitor, mock_timer_manager
    ):
        """Test exception handling in expired timer checking."""
        # Setup
        mock_timer_manager.get_expiring_timers.side_effect = Exception("Redis error")

        # Execute
        expired_timers = await timer_monitor.check_expired_timers()

        # Verify
        assert len(expired_timers) == 0

    @pytest.mark.asyncio
    async def test_process_expired_timer_success(
        self, timer_monitor, mock_timer_manager, mock_message_bus, sample_timer
    ):
        """Test successful expired timer processing."""
        # Setup
        mock_timer_manager.update_timer_status.return_value = True

        # Execute
        result = await timer_monitor.process_expired_timer(sample_timer)

        # Verify
        assert result is True
        mock_timer_manager.update_timer_status.assert_called_with(
            "timer_123", "completed"
        )
        mock_message_bus.publish.assert_called()

        # Check that TIMER_EXPIRED event was published
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][1] == MessageType.TIMER_EXPIRED

    @pytest.mark.asyncio
    async def test_process_expired_timer_recurring(
        self, timer_monitor, mock_timer_manager, mock_message_bus, sample_timer
    ):
        """Test processing expired recurring timer."""
        # Setup
        sample_timer["type"] = "recurring"
        sample_timer["recurring"] = {"pattern": "daily"}
        mock_timer_manager.update_timer_status.return_value = True
        mock_timer_manager.create_recurring_instance.return_value = "timer_456"

        # Execute
        result = await timer_monitor.process_expired_timer(sample_timer)

        # Verify
        assert result is True
        mock_timer_manager.create_recurring_instance.assert_called_with(sample_timer)
        mock_message_bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_process_expired_timer_exception(
        self, timer_monitor, mock_timer_manager, sample_timer
    ):
        """Test exception handling in expired timer processing."""
        # Setup
        mock_timer_manager.update_timer_status.side_effect = Exception("Update failed")

        # Execute
        result = await timer_monitor.process_expired_timer(sample_timer)

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_process_timer_actions_notify(self, timer_monitor, sample_timer):
        """Test processing notification actions."""
        # Setup
        sample_timer["actions"] = [{"type": "notify", "config": {"sound": True}}]

        # Execute
        await timer_monitor._process_timer_actions(sample_timer)

        # Verify - should not raise exception
        assert True  # If we get here, the action was processed

    @pytest.mark.asyncio
    async def test_process_timer_actions_request_trigger(
        self, timer_monitor, mock_message_bus, sample_timer
    ):
        """Test processing request trigger actions."""
        # Setup
        sample_timer["actions"] = [
            {
                "type": "trigger_request",
                "request": "Turn on lights",
                "context": {"room": "living_room"},
            }
        ]

        # Execute
        await timer_monitor._process_timer_actions(sample_timer)

        # Verify
        mock_message_bus.publish.assert_called()
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][1] == MessageType.INCOMING_REQUEST
        assert "Turn on lights" in str(call_args[0][2])

    @pytest.mark.asyncio
    async def test_process_timer_actions_webhook(self, timer_monitor, sample_timer):
        """Test processing webhook actions."""
        # Setup
        sample_timer["actions"] = [
            {"type": "webhook", "url": "https://example.com/webhook", "method": "POST"}
        ]

        # Mock aiohttp - patch the import inside the function
        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_client_session.return_value.__aenter__.return_value = mock_session

            # Execute
            await timer_monitor._process_timer_actions(sample_timer)

            # Verify
            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_timer_actions_unknown_type(
        self, timer_monitor, sample_timer
    ):
        """Test processing unknown action types."""
        # Setup
        sample_timer["actions"] = [{"type": "unknown_action"}]

        # Execute
        await timer_monitor._process_timer_actions(sample_timer)

        # Verify - should not raise exception
        assert True

    @pytest.mark.asyncio
    async def test_process_webhook_action_missing_url(
        self, timer_monitor, sample_timer
    ):
        """Test webhook action with missing URL."""
        # Setup
        action = {"type": "webhook", "method": "POST"}

        # Execute
        await timer_monitor._process_webhook_action(sample_timer, action)

        # Verify - should not raise exception
        assert True

    @pytest.mark.asyncio
    async def test_process_webhook_action_get_method(self, timer_monitor, sample_timer):
        """Test webhook action with GET method."""
        # Setup
        action = {
            "type": "webhook",
            "url": "https://example.com/webhook",
            "method": "GET",
        }

        # Mock aiohttp
        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_client_session.return_value.__aenter__.return_value = mock_session

            # Execute
            await timer_monitor._process_webhook_action(sample_timer, action)

            # Verify
            mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_webhook_action_exception(self, timer_monitor, sample_timer):
        """Test webhook action with network exception."""
        # Setup
        action = {
            "type": "webhook",
            "url": "https://example.com/webhook",
            "method": "POST",
        }

        # Mock aiohttp to raise exception
        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_client_session.side_effect = Exception("Network error")

            # Execute
            await timer_monitor._process_webhook_action(sample_timer, action)

            # Verify - should not raise exception
            assert True

    def test_publish_timer_expired_event(
        self, timer_monitor, mock_message_bus, sample_timer
    ):
        """Test publishing timer expired event."""
        # Execute
        timer_monitor._publish_timer_expired_event(sample_timer, "next_timer_456")

        # Verify
        mock_message_bus.publish.assert_called_once()
        call_args = mock_message_bus.publish.call_args
        assert call_args[0][1] == MessageType.TIMER_EXPIRED

        event_data = call_args[0][2]
        assert event_data["timer_id"] == "timer_123"
        assert event_data["next_timer_id"] == "next_timer_456"

    def test_get_monitoring_stats(self, timer_monitor):
        """Test getting monitoring statistics."""
        # Setup
        timer_monitor.processing_timers.add("timer_123")
        timer_monitor.last_check = 1234567890

        # Execute
        stats = timer_monitor.get_monitoring_stats()

        # Verify
        assert stats["last_check"] == 1234567890
        assert stats["check_interval"] == 5
        assert stats["processing_timers_count"] == 1
        assert "timer_123" in stats["processing_timer_ids"]

    @pytest.mark.asyncio
    async def test_cleanup_stale_processing_timers(self, timer_monitor):
        """Test cleanup of stale processing timers."""
        # Setup - add many processing timers to trigger cleanup
        for i in range(150):
            timer_monitor.processing_timers.add(f"timer_{i}")

        # Execute
        await timer_monitor.cleanup_stale_processing_timers()

        # Verify - should clear the set
        assert len(timer_monitor.processing_timers) == 0

    @pytest.mark.asyncio
    async def test_force_check_timers(
        self, timer_monitor, mock_timer_manager, sample_timer
    ):
        """Test force checking timers."""
        # Setup
        timer_monitor.last_check = int(time.time())  # Set recent time
        mock_timer_manager.get_expiring_timers.return_value = [sample_timer]

        # Execute
        expired_timers = await timer_monitor.force_check_timers()

        # Verify - should bypass rate limiting
        assert len(expired_timers) == 1
        mock_timer_manager.get_expiring_timers.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_timers_deduplication(
        self, timer_monitor, mock_timer_manager, sample_timer
    ):
        """Test that timers already being processed are filtered out."""
        # Setup
        timer_monitor.processing_timers.add("timer_123")
        mock_timer_manager.get_expiring_timers.return_value = [sample_timer]

        # Execute
        expired_timers = await timer_monitor.check_expired_timers()

        # Verify - should filter out timer already being processed
        assert len(expired_timers) == 0

    @pytest.mark.asyncio
    async def test_concurrent_timer_processing(
        self, timer_monitor, mock_timer_manager, mock_message_bus
    ):
        """Test concurrent processing of multiple timers."""
        # Setup
        timers = [
            {
                "id": f"timer_{i}",
                "name": f"Timer {i}",
                "type": "countdown",
                "status": "active",
            }
            for i in range(3)
        ]
        mock_timer_manager.update_timer_status.return_value = True

        # Execute
        tasks = [timer_monitor.process_expired_timer(timer) for timer in timers]
        results = await asyncio.gather(*tasks)

        # Verify
        assert all(results)
        assert mock_timer_manager.update_timer_status.call_count == 3
        assert mock_message_bus.publish.call_count == 3
