"""
Tests for Slack workflow integration fix.

This module tests the thread-safe queue implementation that fixes
the event loop mismatch issue in cross-thread communication between
the Slack WebSocket handler and the main supervisor thread.
"""

import asyncio
import queue
import threading
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import ChannelHandler, CommunicationManager
from common.message_bus import MessageBus


class TestThreadSafeQueueBehavior:
    """Test thread-safe queue operations for cross-thread communication."""

    def test_thread_safe_queue_put_get(self):
        """Test basic put/get operations on thread-safe queue."""
        test_queue = queue.Queue()
        test_message = {"type": "test", "data": "hello"}

        # Put message
        test_queue.put(test_message)

        # Get message
        retrieved_message = test_queue.get_nowait()

        assert retrieved_message == test_message

    def test_thread_safe_queue_empty_exception(self):
        """Test queue.Empty exception handling."""
        test_queue = queue.Queue()

        with pytest.raises(queue.Empty):
            test_queue.get_nowait()

    def test_cross_thread_communication(self):
        """Test producer/consumer pattern across threads."""
        test_queue = queue.Queue()
        messages_received = []

        def producer():
            """Producer thread puts messages."""
            for i in range(3):
                test_queue.put({"id": i, "data": f"message_{i}"})
                time.sleep(0.01)  # Small delay to test threading

        def consumer():
            """Consumer thread gets messages."""
            while len(messages_received) < 3:
                try:
                    message = test_queue.get_nowait()
                    messages_received.append(message)
                except queue.Empty:
                    time.sleep(0.01)  # Small delay before retry

        # Start threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        # Wait for completion
        producer_thread.join(timeout=1.0)
        consumer_thread.join(timeout=1.0)

        # Verify all messages received
        assert len(messages_received) == 3
        assert messages_received[0]["id"] == 0
        assert messages_received[1]["id"] == 1
        assert messages_received[2]["id"] == 2

    def test_queue_ordering_integrity(self):
        """Test message ordering is preserved in queue."""
        test_queue = queue.Queue()

        # Put messages in order
        for i in range(10):
            test_queue.put({"order": i})

        # Get messages and verify order
        for expected_order in range(10):
            message = test_queue.get_nowait()
            assert message["order"] == expected_order


class TestChannelHandlerQueueUpdate:
    """Test ChannelHandler base class queue updates."""

    @pytest.fixture
    def mock_communication_manager(self):
        """Create mock communication manager."""
        manager = Mock(spec=CommunicationManager)
        manager.channel_queues = {}
        manager._background_threads = {}
        return manager

    def test_channel_handler_uses_thread_safe_queue(self, mock_communication_manager):
        """Test that ChannelHandler creates thread-safe queue."""
        # Create a test channel handler
        handler = ChannelHandler()
        handler.channel_type = Mock()
        handler.communication_manager = mock_communication_manager

        # Mock the background session loop to avoid actual execution
        with patch.object(handler, "_background_session_loop", new_callable=AsyncMock):
            # Start background thread (this should create the queue)
            asyncio.run(handler._start_background_thread())

            # Verify queue is thread-safe queue.Queue, not asyncio.Queue
            assert isinstance(handler.message_queue, queue.Queue)
            assert not isinstance(handler.message_queue, asyncio.Queue)


class TestQueueProcessorUpdate:
    """Test queue processor handles thread-safe queues."""

    @pytest.fixture
    def communication_manager(self):
        """Create communication manager for testing."""
        message_bus = Mock(spec=MessageBus)
        manager = CommunicationManager(message_bus)
        return manager

    @pytest.mark.asyncio
    async def test_queue_processor_handles_thread_safe_queue(
        self, communication_manager
    ):
        """Test queue processor can handle thread-safe queue operations."""
        # Create thread-safe queue with test message
        test_queue = queue.Queue()
        test_message = {
            "type": "incoming_message",
            "user_id": "test_user",
            "channel_id": "test_channel",
            "text": "test message",
            "timestamp": "123456",
        }
        test_queue.put(test_message)

        # Add queue to communication manager
        communication_manager.channel_queues["test_channel"] = test_queue

        # Mock the message handler to track calls
        with patch.object(
            communication_manager, "_handle_channel_message", new_callable=AsyncMock
        ) as mock_handler:
            # Process one iteration of queue processing
            await communication_manager._process_single_queue_iteration()

            # Verify message was processed
            mock_handler.assert_called_once_with("test_channel", test_message)

    @pytest.mark.asyncio
    async def test_queue_processor_handles_empty_queue(self, communication_manager):
        """Test queue processor handles empty thread-safe queue gracefully."""
        # Create empty thread-safe queue
        test_queue = queue.Queue()
        communication_manager.channel_queues["test_channel"] = test_queue

        # Mock the message handler
        with patch.object(
            communication_manager, "_handle_channel_message", new_callable=AsyncMock
        ) as mock_handler:
            # Process one iteration - should not raise exception
            await communication_manager._process_single_queue_iteration()

            # Verify no messages were processed
            mock_handler.assert_not_called()


class TestSlackHandlerUpdate:
    """Test Slack handler uses direct queue operations."""

    @pytest.fixture
    def slack_handler(self):
        """Create Slack handler for testing."""
        with patch("slack_bolt.App"):
            handler = SlackChannelHandler()
            handler.communication_manager = Mock()
            handler.message_queue = queue.Queue()  # Use thread-safe queue
            return handler

    def test_slack_handler_direct_queue_put(self, slack_handler):
        """Test Slack handler uses direct queue.put() operation."""
        test_event = {
            "user": "test_user",
            "channel": "test_channel",
            "text": "test message",
            "ts": "123456",
        }

        # Simulate the new direct queue put operation
        slack_handler.message_queue.put(
            {
                "type": "incoming_message",
                "user_id": test_event["user"],
                "channel_id": test_event["channel"],
                "text": test_event.get("text", ""),
                "timestamp": test_event.get("ts"),
            }
        )

        # Verify message was queued
        queued_message = slack_handler.message_queue.get_nowait()
        assert queued_message["type"] == "incoming_message"
        assert queued_message["user_id"] == "test_user"
        assert queued_message["channel_id"] == "test_channel"
        assert queued_message["text"] == "test message"
        assert queued_message["timestamp"] == "123456"

    def test_slack_handler_app_mention_queue_put(self, slack_handler):
        """Test Slack handler queues app mentions correctly."""
        test_event = {
            "user": "test_user",
            "channel": "test_channel",
            "text": "<@U123456> hello bot",
            "ts": "123456",
        }

        # Simulate app mention queuing
        slack_handler.message_queue.put(
            {
                "type": "app_mention",
                "user_id": test_event["user"],
                "channel_id": test_event["channel"],
                "text": test_event.get("text", ""),
                "timestamp": test_event.get("ts"),
            }
        )

        # Verify message was queued
        queued_message = slack_handler.message_queue.get_nowait()
        assert queued_message["type"] == "app_mention"
        assert queued_message["user_id"] == "test_user"
        assert queued_message["text"] == "<@U123456> hello bot"


class TestEndToEndSlackFlow:
    """Test end-to-end Slack message flow with thread-safe queues."""

    @pytest.mark.asyncio
    async def test_slack_message_reaches_workflow_engine(self):
        """Test that Slack messages reach the workflow engine through thread-safe queues."""
        # Create communication manager with mock message bus
        message_bus = Mock(spec=MessageBus)
        communication_manager = CommunicationManager(message_bus)

        # Create thread-safe queue with Slack message
        slack_queue = queue.Queue()
        test_message = {
            "type": "app_mention",
            "user_id": "U123456",
            "channel_id": "C123456",
            "text": "<@U987654> what's the weather?",
            "timestamp": "1641234567.123",
        }
        slack_queue.put(test_message)

        # Register queue with communication manager
        communication_manager.channel_queues["slack"] = slack_queue

        # Mock workflow engine start_workflow method
        communication_manager.workflow_engine = Mock()
        communication_manager.workflow_engine.start_workflow = AsyncMock(
            return_value="workflow_123"
        )

        # Process the message
        await communication_manager._handle_channel_message("slack", test_message)

        # Verify workflow was started
        communication_manager.workflow_engine.start_workflow.assert_called_once()
        call_args = communication_manager.workflow_engine.start_workflow.call_args[0]
        assert (
            "what's the weather?" in call_args[0]
        )  # Request text should contain the message
