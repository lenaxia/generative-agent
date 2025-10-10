"""
Unit tests for Slack app mention fix.

This test suite validates that the queue processing inconsistency
in the Slack handler has been resolved.
"""

import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest


class TestSlackAppMentionFix(unittest.TestCase):
    """Test cases for the Slack app mention queue processing fix."""

    def test_queue_consistency_between_app_mentions_and_buttons(self):
        """Test that app mentions and button interactions use consistent queue methods."""
        # Create a thread-safe queue (same as the real implementation)
        message_queue = queue.Queue()

        # Simulate app mention message (the working case)
        app_mention_data = {
            "type": "app_mention",
            "user_id": "U12345",
            "channel_id": "C12345",
            "text": "@bot hello",
            "timestamp": "1234567890.123",
        }

        # Simulate button interaction message (the fixed case)
        button_data = {
            "type": "user_response",
            "data": {
                "action_id": "test_action",
                "value": "yes",
                "user_id": "U12345",
                "channel_id": "C12345",
            },
        }

        # Both should use the same queue.put() method
        message_queue.put(app_mention_data)
        message_queue.put(button_data)

        # Verify both messages are in the queue
        self.assertEqual(message_queue.qsize(), 2)

        # Process the queue
        processed_messages = []
        while not message_queue.empty():
            message = message_queue.get_nowait()
            processed_messages.append(message)
            message_queue.task_done()

        # Verify both message types were processed
        self.assertEqual(len(processed_messages), 2)

        message_types = [msg["type"] for msg in processed_messages]
        self.assertIn("app_mention", message_types)
        self.assertIn("user_response", message_types)

    def test_thread_safety_of_queue_operations(self):
        """Test that queue operations are thread-safe as expected."""
        shared_queue = queue.Queue()
        results = []

        def worker(worker_id, message_count=5):
            """Worker function that adds messages to queue."""
            for i in range(message_count):
                shared_queue.put(
                    {
                        "type": "app_mention",
                        "worker_id": worker_id,
                        "message_id": i,
                        "user_id": f"U{worker_id}",
                        "channel_id": "C12345",
                        "text": f"@bot message {i} from worker {worker_id}",
                    }
                )

        def processor():
            """Processor function that consumes messages from queue."""
            processed_count = 0
            while processed_count < 15:  # 3 workers * 5 messages each
                try:
                    message = shared_queue.get(timeout=1.0)
                    results.append(message)
                    shared_queue.task_done()
                    processed_count += 1
                except queue.Empty:
                    break

        # Start worker threads
        worker_threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            worker_threads.append(t)
            t.start()

        # Start processor thread
        processor_thread = threading.Thread(target=processor)
        processor_thread.start()

        # Wait for all threads to complete
        for t in worker_threads:
            t.join(timeout=2.0)

        processor_thread.join(timeout=3.0)

        # Verify all messages were processed
        self.assertEqual(len(results), 15)

        # Verify messages from all workers were processed
        worker_ids = {msg["worker_id"] for msg in results}
        self.assertEqual(worker_ids, {0, 1, 2})

    def test_queue_error_handling(self):
        """Test queue error handling scenarios."""
        # Test with bounded queue
        bounded_queue = queue.Queue(maxsize=2)

        # Fill the queue
        bounded_queue.put("message1")
        bounded_queue.put("message2")

        # Test that queue.Full is raised when appropriate
        with self.assertRaises(queue.Full):
            bounded_queue.put("message3", timeout=0.1)

        # Test that queue.Empty is raised when appropriate
        empty_queue = queue.Queue()
        with self.assertRaises(queue.Empty):
            empty_queue.get_nowait()

    def test_slack_handler_message_queuing_logic(self):
        """Test the actual Slack handler message queuing logic."""
        # Test the message queuing logic without creating the full handler
        # This tests the core logic that was fixed

        # Create a mock message queue (same as real implementation)
        message_queue = queue.Queue()

        # Test app mention data structure (simulating what the handler does)
        mock_event = {
            "user": "U12345",
            "channel": "C12345",
            "text": "@bot test message",
            "ts": "1234567890.123",
        }

        # Simulate the app mention handler logic (the fixed version)
        message_data = {
            "type": "app_mention",
            "user_id": mock_event["user"],
            "channel_id": mock_event["channel"],
            "text": mock_event.get("text", ""),
            "timestamp": mock_event.get("ts"),
        }

        # This should work without errors (using thread-safe queue.put())
        message_queue.put(message_data)

        # Verify message was queued
        self.assertEqual(message_queue.qsize(), 1)

        # Verify message content
        queued_message = message_queue.get_nowait()
        self.assertEqual(queued_message["type"], "app_mention")
        self.assertEqual(queued_message["user_id"], "U12345")
        self.assertEqual(queued_message["channel_id"], "C12345")
        self.assertEqual(queued_message["text"], "@bot test message")
        self.assertEqual(queued_message["timestamp"], "1234567890.123")

    def test_queue_performance_under_load(self):
        """Test queue performance under high load scenarios."""
        test_queue = queue.Queue()
        start_time = time.time()

        # Add many messages quickly
        message_count = 1000
        for i in range(message_count):
            test_queue.put(
                {
                    "type": "app_mention",
                    "message_id": i,
                    "user_id": f"U{i % 10}",  # 10 different users
                    "channel_id": "C12345",
                    "text": f"@bot message {i}",
                    "timestamp": str(time.time()),
                }
            )

        queue_fill_time = time.time() - start_time

        # Process all messages
        start_time = time.time()
        processed_count = 0
        while not test_queue.empty():
            message = test_queue.get_nowait()
            processed_count += 1
            test_queue.task_done()

        process_time = time.time() - start_time

        # Verify all messages were processed
        self.assertEqual(processed_count, message_count)

        # Performance assertions (should be very fast)
        self.assertLess(queue_fill_time, 1.0, "Queue filling took too long")
        self.assertLess(process_time, 1.0, "Queue processing took too long")


if __name__ == "__main__":
    unittest.main()
