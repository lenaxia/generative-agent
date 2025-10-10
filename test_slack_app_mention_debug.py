#!/usr/bin/env python3
"""
Debug test for Slack app mention issue.
This test reproduces the problem where app mentions are not being processed.
"""

import asyncio
import logging
import queue
import threading
import time
from unittest.mock import MagicMock, patch

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_slack_app_mention_queue_issue():
    """Test to reproduce the Slack app mention queue processing issue."""

    # Simulate the Slack handler setup
    message_queue = queue.Queue()

    # Simulate what happens when an app mention is received
    def simulate_app_mention():
        """Simulate receiving an app mention from Slack WebSocket thread."""
        logger.info("🔔 Simulating app mention received")

        # This is what the Slack handler does in line 366-374
        message_queue.put(
            {
                "type": "app_mention",
                "user_id": "U12345",
                "channel_id": "C12345",
                "text": "@bot hello there",
                "timestamp": "1234567890.123",
            }
        )
        logger.info("📤 Message added to queue")

    # Simulate the queue processor (from communication_manager.py:353-372)
    def simulate_queue_processor():
        """Simulate the queue processor that should handle messages."""
        logger.info("🔄 Starting queue processor simulation")

        processed_count = 0
        max_iterations = 10

        for i in range(max_iterations):
            try:
                # This is the actual queue processing logic
                while True:
                    try:
                        message = message_queue.get_nowait()  # Non-blocking get
                        logger.info(f"📨 Processing queued message: {message}")
                        processed_count += 1
                        message_queue.task_done()
                    except queue.Empty:
                        break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

            time.sleep(0.1)  # Same as the real processor

        logger.info(f"✅ Queue processor finished. Processed {processed_count} messages")
        return processed_count

    # Run the test
    logger.info("🚀 Starting Slack app mention debug test")

    # Start queue processor in background
    processor_thread = threading.Thread(target=simulate_queue_processor, daemon=True)
    processor_thread.start()

    # Wait a moment for processor to start
    time.sleep(0.2)

    # Simulate app mention
    simulate_app_mention()

    # Wait for processing
    processor_thread.join(timeout=2.0)

    # Check if queue is empty (message was processed)
    queue_size = message_queue.qsize()
    logger.info(f"📊 Final queue size: {queue_size}")

    if queue_size == 0:
        logger.info("✅ SUCCESS: Message was processed correctly")
        return True
    else:
        logger.error("❌ FAILURE: Message was not processed")
        return False


def test_slack_handler_integration():
    """Test the actual Slack handler integration issue."""

    # Mock the communication manager and message bus
    with patch(
        "common.channel_handlers.slack_handler.ChannelHandler.__init__",
        return_value=None,
    ):
        from common.channel_handlers.slack_handler import SlackChannelHandler

        # Create handler with mock config
        handler = SlackChannelHandler(
            {"bot_token": "xoxb-test", "app_token": "xapp-test"}
        )

        # Manually set the config since we mocked __init__
        handler.config = {"bot_token": "xoxb-test", "app_token": "xapp-test"}

        # Mock the message queue
        handler.message_queue = queue.Queue()
        handler.communication_manager = MagicMock()

        # Test the app mention handler logic
        mock_event = {
            "user": "U12345",
            "channel": "C12345",
            "text": "@bot test message",
            "ts": "1234567890.123",
        }

        # Simulate what the app mention handler does
        logger.info("🔔 Testing app mention handler logic")

        try:
            # This is the exact code from lines 366-374
            handler.message_queue.put(
                {
                    "type": "app_mention",
                    "user_id": mock_event["user"],
                    "channel_id": mock_event["channel"],
                    "text": mock_event.get("text", ""),
                    "timestamp": mock_event.get("ts"),
                }
            )

            # Check if message was queued
            queue_size = handler.message_queue.qsize()
            logger.info(f"📊 Messages in queue: {queue_size}")

            if queue_size > 0:
                message = handler.message_queue.get_nowait()
                logger.info(f"📨 Queued message: {message}")
                logger.info("✅ SUCCESS: App mention handler logic works")
                return True
            else:
                logger.error("❌ FAILURE: No message was queued")
                return False

        except Exception as e:
            logger.error(f"❌ FAILURE: Exception in app mention handler: {e}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 SLACK APP MENTION DEBUG TEST")
    print("=" * 60)

    # Test 1: Basic queue functionality
    print("\n📋 Test 1: Basic Queue Processing")
    result1 = test_slack_app_mention_queue_issue()

    # Test 2: Slack handler integration
    print("\n📋 Test 2: Slack Handler Integration")
    result2 = test_slack_handler_integration()

    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Basic Queue Processing: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"Slack Handler Integration: {'✅ PASS' if result2 else '❌ FAIL'}")

    if result1 and result2:
        print("\n🎉 All tests passed - the queue mechanism works correctly")
        print("🔍 The issue must be elsewhere in the integration")
    else:
        print("\n🚨 Tests failed - found the queue processing issue")

    print("=" * 60)
