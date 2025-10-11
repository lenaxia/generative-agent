"""
Integration tests for Timer Communication System.

Tests the integration between the Timer System, Communication Manager, and Channel Handlers.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.communication_manager import (
    ChannelType,
    CommunicationManager,
    DeliveryGuarantee,
    MessageFormat,
)
from common.message_bus import MessageBus, MessageType
from roles.timer.lifecycle import TimerManager
from supervisor.timer_monitor import TimerMonitor


class TestTimerCommunicationIntegration(unittest.TestCase):
    """Integration tests for Timer Communication System."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock Redis
        self.redis_patcher = patch("redis.Redis")
        self.mock_redis = self.redis_patcher.start()

        # Mock Redis instance
        self.mock_redis_instance = MagicMock()
        self.mock_redis.return_value = self.mock_redis_instance

        # Mock Redis methods
        self.mock_redis_instance.hgetall.return_value = {}
        self.mock_redis_instance.zadd.return_value = 1
        self.mock_redis_instance.hmset.return_value = True
        self.mock_redis_instance.zrangebyscore.return_value = []

        # Set up MessageBus
        self.message_bus = MessageBus()

        # Set up TimerManager with mocked Redis
        self.timer_manager = TimerManager(redis_host="localhost")

        # Set up TimerMonitor
        self.timer_monitor = TimerMonitor(self.message_bus)

        # Patch the global timer manager
        with patch("roles.timer.lifecycle._timer_manager", self.timer_manager):
            # Set up CommunicationManager
            self.communication_manager = CommunicationManager.get_instance()

            # Replace the message bus in the communication manager
            self.communication_manager.message_bus = self.message_bus

            # Register mock message handler
            self.message_bus.subscribe(
                self.communication_manager,
                MessageType.TIMER_EXPIRED,
                self.communication_manager._handle_timer_expired,
            )

    def tearDown(self):
        """Tear down test fixtures."""
        self.redis_patcher.stop()

        # Reset singletons
        CommunicationManager._instance = None
        MessageBus._instance = None
        TimerManager._instance = None

    @pytest.mark.asyncio
    async def test_timer_expiry_triggers_notification(self):
        """Test that timer expiry triggers notification via Communication Manager."""
        # Create a mock channel handler
        mock_handler = AsyncMock()
        mock_handler.send_notification = AsyncMock(return_value={"success": True})
        mock_handler.channel_type = ChannelType.CONSOLE

        # Register the mock handler
        self.communication_manager.channels[ChannelType.CONSOLE] = mock_handler

        # Create a timer with notification metadata
        timer_id = "test_timer_123"
        timer_data = {
            "id": timer_id,
            "name": "Test Timer",
            "type": "countdown",
            "status": "active",
            "created_at": 1000000000,
            "expires_at": 1000000060,  # 60 seconds later
            "duration_seconds": 60,
            "user_id": "test_user",
            "channel_id": "test_channel",
            "custom_message": "Timer test message",
            "metadata": {
                "notification_channel": "console",
                "notification_recipient": "test_user",
                "notification_guarantee": "at_least_once",
            },
        }

        # Mock the timer retrieval
        self.mock_redis_instance.zrangebyscore.return_value = [timer_id.encode()]
        self.mock_redis_instance.hgetall.return_value = {
            k.encode(): (
                json.dumps(v).encode()
                if isinstance(v, (dict, list))
                else str(v).encode()
            )
            for k, v in timer_data.items()
        }

        # Check for expired timers
        expired_timers = await self.timer_monitor.check_expired_timers()
        self.assertEqual(len(expired_timers), 1)

        # Process the expired timer
        await self.timer_monitor.process_expired_timer(expired_timers[0])

        # Verify the notification was sent
        mock_handler.send_notification.assert_called_once()

        # Check notification content
        call_args = mock_handler.send_notification.call_args[0]
        self.assertIn("Timer expired", call_args[0])  # Message contains "Timer expired"
        self.assertEqual(call_args[1], "test_user")  # Recipient is correct

    @pytest.mark.asyncio
    async def test_notification_to_multiple_channels(self):
        """Test notification sent to multiple channels in parallel."""
        # Create mock channel handlers
        mock_slack = AsyncMock()
        mock_slack.send_notification = AsyncMock(
            return_value={"success": False, "error": "Connection failed"}
        )
        mock_slack.channel_type = ChannelType.SLACK

        mock_console = AsyncMock()
        mock_console.send_notification = AsyncMock(return_value={"success": True})
        mock_console.channel_type = ChannelType.CONSOLE

        # Register the mock handlers
        self.communication_manager.channels[ChannelType.SLACK] = mock_slack
        self.communication_manager.channels[ChannelType.CONSOLE] = mock_console

        # Create a timer with Slack notification channel
        timer_data = {
            "data": {
                "timer_id": "test_timer_456",
                "name": "Test Timer",
                "notification_channel": "slack",
                "notification_recipient": "#test-channel",
                "notification_guarantee": "at_least_once",
            }
        }

        # Simulate timer expired event
        await self.communication_manager._handle_timer_expired(timer_data)

        # Verify both channels were attempted
        mock_slack.send_notification.assert_called_once()
        mock_console.send_notification.assert_called_once()

    @pytest.mark.asyncio
    async def test_at_least_once_notification_tries_all_channels(self):
        """Test that at_least_once notifications try all available channels."""
        # Create mock channel handlers
        mock_slack = AsyncMock()
        mock_slack.send_notification = AsyncMock(
            return_value={"success": False, "error": "Connection failed"}
        )
        mock_slack.channel_type = ChannelType.SLACK

        mock_email = AsyncMock()
        mock_email.send_notification = AsyncMock(
            return_value={"success": False, "error": "Email server down"}
        )
        mock_email.channel_type = ChannelType.EMAIL

        mock_console = AsyncMock()
        mock_console.send_notification = AsyncMock(return_value={"success": True})
        mock_console.channel_type = ChannelType.CONSOLE

        # Register the mock handlers
        self.communication_manager.channels[ChannelType.SLACK] = mock_slack
        self.communication_manager.channels[ChannelType.EMAIL] = mock_email
        self.communication_manager.channels[ChannelType.CONSOLE] = mock_console

        # Create a timer with Slack notification channel and critical priority
        timer_data = {
            "data": {
                "timer_id": "test_timer_789",
                "name": "Critical Timer",
                "notification_channel": "slack",
                "notification_recipient": "#alerts",
                "notification_guarantee": "at_least_once",
            }
        }

        # Simulate timer expired event
        await self.communication_manager._handle_timer_expired(timer_data)

        # Verify all channels were attempted in parallel
        mock_slack.send_notification.assert_called_once()
        mock_email.send_notification.assert_called_once()
        mock_console.send_notification.assert_called_once()


if __name__ == "__main__":
    unittest.main()
