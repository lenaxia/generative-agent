"""Integration tests for timer notification threading fix.

Tests end-to-end timer expiry notification delivery to ensure the
threading fix resolves the issue where timer notifications hang
when sent from background threads.
"""

import asyncio
import threading
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests

from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import (
    ChannelType,
    CommunicationManager,
    MessageFormat,
)
from supervisor.heartbeat import Heartbeat
from supervisor.timer_monitor import TimerMonitor


class TestTimerNotificationFix:
    """Integration test suite for timer notification threading fix."""

    @pytest.fixture
    def mock_slack_config(self):
        """Mock Slack configuration for testing."""
        return {
            "slack": {
                "bot_token": "xoxb-test-token",
                "default_channel": "#general",
            }
        }

    @pytest.fixture
    def communication_manager(self, mock_slack_config):
        """Create CommunicationManager with mocked dependencies."""
        from common.message_bus import MessageBus

        # Mock MessageBus
        mock_message_bus = Mock(spec=MessageBus)
        mock_message_bus.subscribe = Mock()

        # Create CommunicationManager with mocked MessageBus
        with patch(
            "common.communication_manager.MessageBus", return_value=mock_message_bus
        ):
            comm_manager = CommunicationManager(mock_slack_config)

        # Mock the Slack handler to avoid real API calls
        mock_slack_handler = Mock(spec=SlackChannelHandler)
        mock_slack_handler.channel_type = ChannelType.SLACK
        mock_slack_handler.enabled = True

        # Replace the actual handler with our mock
        comm_manager.handlers[ChannelType.SLACK] = mock_slack_handler

        return comm_manager

    @pytest.mark.asyncio
    async def test_timer_expiry_notification_delivery(self, communication_manager):
        """Test end-to-end timer expiry notification delivery."""
        # Mock successful Slack API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "ts": "1234567890.123456",
            "channel": "#general",
        }

        # Track notification calls
        notification_calls = []

        async def mock_send_notification(message, recipient, message_format, metadata):
            notification_calls.append(
                {
                    "message": message,
                    "recipient": recipient,
                    "format": message_format,
                    "metadata": metadata,
                    "thread": threading.current_thread().name,
                }
            )
            return {"success": True, "channel": recipient, "ts": "1234567890.123456"}

        # Mock the handler's send_notification method
        slack_handler = communication_manager.handlers[ChannelType.SLACK]
        slack_handler.send_notification = mock_send_notification

        # Simulate timer expiry notification from background thread
        def simulate_timer_expiry():
            """Simulate timer expiry in background thread (like TimerMonitor)."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # This simulates what happens when a timer expires
                result = loop.run_until_complete(
                    communication_manager.send_notification(
                        message="⏰ 10s timer expired!",
                        channels=[ChannelType.SLACK],
                        recipient="#general",
                        message_format=MessageFormat.PLAIN_TEXT,
                        metadata={"timer_id": "test_timer_123"},
                    )
                )
                notification_calls.append({"result": result})
            finally:
                loop.close()

        # Run timer expiry simulation in background thread
        timer_thread = threading.Thread(
            target=simulate_timer_expiry, name="TimerMonitorThread"
        )
        timer_thread.start()
        timer_thread.join()

        # Verify notification was delivered
        assert len(notification_calls) >= 1

        # Find the actual notification call (not the result)
        notification_call = next(
            (call for call in notification_calls if "message" in call), None
        )
        assert notification_call is not None

        # Verify notification content
        assert "⏰ 10s timer expired!" in notification_call["message"]
        assert notification_call["recipient"] == "#general"
        assert (
            notification_call["thread"] == "TimerMonitorThread"
        )  # Confirms it ran in background thread

    @pytest.mark.asyncio
    async def test_concurrent_timer_notifications(self, communication_manager):
        """Test multiple timer notifications simultaneously."""
        notification_calls = []

        async def mock_send_notification(message, recipient, message_format, metadata):
            # Add small delay to simulate real API call
            await asyncio.sleep(0.1)
            notification_calls.append(
                {
                    "message": message,
                    "recipient": recipient,
                    "thread": threading.current_thread().name,
                    "timestamp": time.time(),
                }
            )
            return {"success": True, "channel": recipient, "ts": f"{time.time():.6f}"}

        slack_handler = communication_manager.handlers[ChannelType.SLACK]
        slack_handler.send_notification = mock_send_notification

        def simulate_multiple_timers():
            """Simulate multiple timer expiries concurrently."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Simulate 3 timers expiring at the same time
                tasks = []
                for i in range(3):
                    task = communication_manager.send_notification(
                        message=f"⏰ Timer {i+1} expired!",
                        channels=[ChannelType.SLACK],
                        recipient="#general",
                        message_format=MessageFormat.PLAIN_TEXT,
                        metadata={"timer_id": f"timer_{i+1}"},
                    )
                    tasks.append(task)

                # Wait for all notifications to complete
                loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                loop.close()

        # Run concurrent timer simulation
        timer_thread = threading.Thread(
            target=simulate_multiple_timers, name="ConcurrentTimerThread"
        )
        timer_thread.start()
        timer_thread.join()

        # Verify all notifications were delivered
        assert len(notification_calls) == 3

        # Verify all ran in background thread
        for call in notification_calls:
            assert call["thread"] == "ConcurrentTimerThread"
            assert "Timer" in call["message"]
            assert "expired" in call["message"]

    @pytest.mark.asyncio
    async def test_thread_safety_under_load(self, communication_manager):
        """Test thread safety with high notification volume."""
        notification_calls = []
        errors = []

        async def mock_send_notification(message, recipient, message_format, metadata):
            try:
                # Simulate variable response times
                await asyncio.sleep(0.05 + (hash(message) % 10) * 0.01)
                notification_calls.append(
                    {
                        "message": message,
                        "thread": threading.current_thread().name,
                        "success": True,
                    }
                )
                return {"success": True, "channel": recipient}
            except Exception as e:
                errors.append(str(e))
                return {"success": False, "error": str(e)}

        slack_handler = communication_manager.handlers[ChannelType.SLACK]
        slack_handler.send_notification = mock_send_notification

        def high_load_simulation():
            """Simulate high load with many notifications."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Send 20 notifications rapidly
                tasks = []
                for i in range(20):
                    task = communication_manager.send_notification(
                        message=f"⏰ High load timer {i} expired!",
                        channels=[ChannelType.SLACK],
                        recipient="#general",
                        message_format=MessageFormat.PLAIN_TEXT,
                        metadata={"timer_id": f"load_timer_{i}"},
                    )
                    tasks.append(task)

                loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                loop.close()

        # Run high load test
        load_thread = threading.Thread(
            target=high_load_simulation, name="HighLoadThread"
        )
        load_thread.start()
        load_thread.join()

        # Verify all notifications succeeded without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(notification_calls) == 20

        # Verify all notifications were successful
        for call in notification_calls:
            assert call["success"] is True
            assert call["thread"] == "HighLoadThread"

    @pytest.mark.asyncio
    async def test_main_thread_vs_background_thread_behavior(
        self, communication_manager
    ):
        """Test that main thread and background thread both work correctly."""
        notification_calls = []

        async def mock_send_notification(message, recipient, message_format, metadata):
            notification_calls.append(
                {
                    "message": message,
                    "thread": threading.current_thread().name,
                    "is_main_thread": threading.current_thread().name == "MainThread",
                }
            )
            return {"success": True, "channel": recipient}

        slack_handler = communication_manager.handlers[ChannelType.SLACK]
        slack_handler.send_notification = mock_send_notification

        # Test main thread notification
        await communication_manager.send_notification(
            message="Main thread notification",
            channels=[ChannelType.SLACK],
            recipient="#general",
            message_format=MessageFormat.PLAIN_TEXT,
            metadata={"source": "main_thread"},
        )

        # Test background thread notification
        def background_notification():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    communication_manager.send_notification(
                        message="Background thread notification",
                        channels=[ChannelType.SLACK],
                        recipient="#general",
                        message_format=MessageFormat.PLAIN_TEXT,
                        metadata={"source": "background_thread"},
                    )
                )
            finally:
                loop.close()

        bg_thread = threading.Thread(
            target=background_notification, name="BackgroundTestThread"
        )
        bg_thread.start()
        bg_thread.join()

        # Verify both notifications worked
        assert len(notification_calls) == 2

        main_call = next(call for call in notification_calls if call["is_main_thread"])
        bg_call = next(
            call for call in notification_calls if not call["is_main_thread"]
        )

        assert "Main thread" in main_call["message"]
        assert "Background thread" in bg_call["message"]
        assert bg_call["thread"] == "BackgroundTestThread"

    @pytest.mark.asyncio
    async def test_timer_notification_with_real_slack_handler(self):
        """Test with actual SlackChannelHandler to verify threading fix."""
        # Create real SlackChannelHandler with test config
        config = {"bot_token": "xoxb-test-token", "default_channel": "#test"}
        slack_handler = SlackChannelHandler(config)

        # Mock the actual HTTP calls to avoid real API requests
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "ts": "1234567890.123456"}

        results = []

        def test_background_thread_call():
            """Test the actual SlackHandler from background thread."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with patch("requests.post", return_value=mock_response):
                    with patch("asyncio.get_event_loop", return_value=loop):
                        # This should use the thread-safe client
                        result = loop.run_until_complete(
                            slack_handler._send_via_api(
                                "⏰ Timer expired from background thread!",
                                "#test",
                                MessageFormat.PLAIN_TEXT,
                                [],
                                {},
                            )
                        )
                        results.append(result)
            finally:
                loop.close()

        # Run in background thread (simulating TimerMonitor)
        timer_thread = threading.Thread(
            target=test_background_thread_call, name="TimerThread"
        )
        timer_thread.start()
        timer_thread.join()

        # Verify the notification succeeded
        assert len(results) == 1
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_error_handling_in_background_thread(self, communication_manager):
        """Test error handling when notifications fail in background threads."""
        error_calls = []

        async def mock_send_notification_with_error(
            message, recipient, message_format, metadata
        ):
            if "fail" in message.lower():
                error_calls.append(
                    {"message": message, "thread": threading.current_thread().name}
                )
                return {"success": False, "error": "Simulated API error"}
            else:
                return {"success": True, "channel": recipient}

        slack_handler = communication_manager.handlers[ChannelType.SLACK]
        slack_handler.send_notification = mock_send_notification_with_error

        def test_error_handling():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Send both successful and failing notifications
                tasks = [
                    communication_manager.send_notification(
                        message="⏰ Success timer expired!",
                        channels=[ChannelType.SLACK],
                        recipient="#general",
                        message_format=MessageFormat.PLAIN_TEXT,
                        metadata={},
                    ),
                    communication_manager.send_notification(
                        message="⏰ FAIL timer expired!",
                        channels=[ChannelType.SLACK],
                        recipient="#general",
                        message_format=MessageFormat.PLAIN_TEXT,
                        metadata={},
                    ),
                ]
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            finally:
                loop.close()

        error_thread = threading.Thread(
            target=test_error_handling, name="ErrorTestThread"
        )
        error_thread.start()
        error_thread.join()

        # Verify error was handled properly
        assert len(error_calls) == 1
        assert "FAIL" in error_calls[0]["message"]
        assert error_calls[0]["thread"] == "ErrorTestThread"

    @pytest.mark.asyncio
    async def test_notification_metadata_preservation(self, communication_manager):
        """Test that notification metadata is preserved through threading fix."""
        received_metadata = []

        async def mock_send_notification(message, recipient, message_format, metadata):
            received_metadata.append(
                {"metadata": metadata, "thread": threading.current_thread().name}
            )
            return {"success": True, "channel": recipient}

        slack_handler = communication_manager.handlers[ChannelType.SLACK]
        slack_handler.send_notification = mock_send_notification

        test_metadata = {
            "timer_id": "test_timer_456",
            "user_id": "U123456789",
            "thread_ts": "1234567890.000000",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Timer notification"},
                }
            ],
        }

        def test_metadata_preservation():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    communication_manager.send_notification(
                        message="⏰ Timer with metadata expired!",
                        channels=[ChannelType.SLACK],
                        recipient="#general",
                        message_format=MessageFormat.RICH_TEXT,
                        metadata=test_metadata,
                    )
                )
            finally:
                loop.close()

        metadata_thread = threading.Thread(
            target=test_metadata_preservation, name="MetadataThread"
        )
        metadata_thread.start()
        metadata_thread.join()

        # Verify metadata was preserved
        assert len(received_metadata) == 1
        preserved_metadata = received_metadata[0]["metadata"]

        assert preserved_metadata["timer_id"] == "test_timer_456"
        assert preserved_metadata["user_id"] == "U123456789"
        assert preserved_metadata["thread_ts"] == "1234567890.000000"
        assert "blocks" in preserved_metadata
        assert received_metadata[0]["thread"] == "MetadataThread"
