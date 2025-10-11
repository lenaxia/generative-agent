"""Unit tests for Slack threading fix implementation.

Tests the thread detection and thread-safe HTTP client functionality
to ensure timer notifications work correctly from background threads.
"""

import asyncio
import threading
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests

from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import MessageFormat


class TestSlackThreadingFix:
    """Test suite for Slack threading fix implementation."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance for testing."""
        config = {
            "bot_token": "xoxb-test-token",
            "default_channel": "#test",
        }
        return SlackChannelHandler(config)

    def test_background_thread_detection_main_thread(self, slack_handler):
        """Test thread context detection in main thread."""
        # Should return False when called from main thread
        assert not slack_handler._is_background_thread()

    def test_background_thread_detection_background_thread(self, slack_handler):
        """Test thread context detection in background thread."""
        result = []

        def background_task():
            result.append(slack_handler._is_background_thread())

        # Run in background thread
        thread = threading.Thread(target=background_task, name="BackgroundThread")
        thread.start()
        thread.join()

        # Should return True when called from background thread
        assert result[0] is True

    @pytest.mark.asyncio
    async def test_main_thread_uses_aiohttp(self, slack_handler):
        """Test that main thread uses aiohttp client."""
        with patch.object(
            slack_handler, "_send_via_api_aiohttp", new_callable=AsyncMock
        ) as mock_aiohttp:
            with patch.object(
                slack_handler, "_send_via_api_threadsafe", new_callable=AsyncMock
            ) as mock_threadsafe:
                mock_aiohttp.return_value = {"success": True}

                await slack_handler._send_via_api(
                    "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
                )

                # Should call aiohttp method, not threadsafe method
                mock_aiohttp.assert_called_once()
                mock_threadsafe.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_thread_uses_threadsafe_client(self, slack_handler):
        """Test that background thread uses thread-safe client."""
        result = []

        async def background_task():
            with patch.object(
                slack_handler, "_send_via_api_aiohttp", new_callable=AsyncMock
            ) as mock_aiohttp:
                with patch.object(
                    slack_handler, "_send_via_api_threadsafe", new_callable=AsyncMock
                ) as mock_threadsafe:
                    mock_threadsafe.return_value = {"success": True}

                    response = await slack_handler._send_via_api(
                        "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
                    )

                    result.append(
                        {
                            "aiohttp_called": mock_aiohttp.called,
                            "threadsafe_called": mock_threadsafe.called,
                            "response": response,
                        }
                    )

        # Run in background thread with event loop
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(background_task())
            finally:
                loop.close()

        thread = threading.Thread(target=run_in_thread, name="BackgroundThread")
        thread.start()
        thread.join()

        # Should call threadsafe method, not aiohttp method
        assert not result[0]["aiohttp_called"]
        assert result[0]["threadsafe_called"]
        assert result[0]["response"]["success"] is True

    @pytest.mark.asyncio
    async def test_thread_safe_http_client_success(self, slack_handler):
        """Test thread-safe HTTP client with successful response."""
        # Mock successful Slack API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "ts": "1234567890.123456",
            "channel": "#test",
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

                result = await slack_handler._send_via_api_threadsafe(
                    "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
                )

                # Verify success response
                assert result["success"] is True
                assert result["channel"] == "#test"
                assert result["ts"] == "1234567890.123456"
                assert result["message_id"] == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_thread_safe_http_client_api_error(self, slack_handler):
        """Test thread-safe HTTP client with Slack API error."""
        # Mock Slack API error response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

                result = await slack_handler._send_via_api_threadsafe(
                    "test message", "#invalid", MessageFormat.PLAIN_TEXT, [], {}
                )

                # Verify error response
                assert result["success"] is False
                assert "channel_not_found" in result["error"]

    @pytest.mark.asyncio
    async def test_thread_safe_http_client_http_error(self, slack_handler):
        """Test thread-safe HTTP client with HTTP error."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch("requests.post", return_value=mock_response) as mock_post:
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

                result = await slack_handler._send_via_api_threadsafe(
                    "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
                )

                # Verify error response
                assert result["success"] is False
                assert "404" in result["error"]
                assert "Not Found" in result["error"]

    @pytest.mark.asyncio
    async def test_thread_safe_http_client_exception(self, slack_handler):
        """Test thread-safe HTTP client with exception."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock(
                side_effect=requests.RequestException("Connection error")
            )

            result = await slack_handler._send_via_api_threadsafe(
                "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
            )

            # Verify error response
            assert result["success"] is False
            assert "Connection error" in result["error"]

    @pytest.mark.asyncio
    async def test_thread_safe_client_with_buttons(self, slack_handler):
        """Test thread-safe client with buttons metadata."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "ts": "1234567890.123456"}

        buttons = [
            {"text": "Yes", "value": "yes", "style": "primary"},
            {"text": "No", "value": "no", "style": "default"},
        ]

        with patch("requests.post", return_value=mock_response) as mock_post:
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

                result = await slack_handler._send_via_api_threadsafe(
                    "test message", "#test", MessageFormat.RICH_TEXT, buttons, {}
                )

                # Verify that blocks were created for buttons
                assert result["success"] is True
                # The executor should have been called with the requests.post function
                mock_loop.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_thread_safe_client_with_metadata(self, slack_handler):
        """Test thread-safe client with various metadata."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "ts": "1234567890.123456"}

        metadata = {
            "thread_ts": "1234567890.000000",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "Custom block"}}
            ],
            "attachments": [{"color": "good", "text": "Attachment text"}],
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

                result = await slack_handler._send_via_api_threadsafe(
                    "test message", "#test", MessageFormat.RICH_TEXT, [], metadata
                )

                # Verify success and that metadata was processed
                assert result["success"] is True
                mock_loop.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_automatic_fallback_selection(self, slack_handler):
        """Test automatic selection between aiohttp and thread-safe clients."""
        results = []

        # Test main thread
        with patch.object(
            slack_handler, "_send_via_api_aiohttp", new_callable=AsyncMock
        ) as mock_aiohttp:
            mock_aiohttp.return_value = {"success": True, "source": "aiohttp"}

            result = await slack_handler._send_via_api(
                "main thread message", "#test", MessageFormat.PLAIN_TEXT, [], {}
            )
            results.append(("main", result))

        # Test background thread
        async def background_test():
            with patch.object(
                slack_handler, "_send_via_api_threadsafe", new_callable=AsyncMock
            ) as mock_threadsafe:
                mock_threadsafe.return_value = {"success": True, "source": "threadsafe"}

                result = await slack_handler._send_via_api(
                    "background thread message",
                    "#test",
                    MessageFormat.PLAIN_TEXT,
                    [],
                    {},
                )
                results.append(("background", result))

        def run_background():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(background_test())
            finally:
                loop.close()

        thread = threading.Thread(target=run_background, name="TestBackgroundThread")
        thread.start()
        thread.join()

        # Verify correct client selection
        assert len(results) == 2
        main_result = next(r for name, r in results if name == "main")
        bg_result = next(r for name, r in results if name == "background")

        assert main_result["success"] is True
        assert bg_result["success"] is True

    def test_thread_detection_with_custom_thread_names(self, slack_handler):
        """Test thread detection with various thread names."""
        test_cases = [
            ("MainThread", False),
            ("BackgroundThread", True),
            ("TimerThread", True),
            ("HeartbeatThread", True),
            ("WorkerThread-1", True),
            ("Thread-1", True),
        ]

        results = []

        def test_thread_name(thread_name, expected):
            def thread_task():
                # Temporarily change thread name
                current_thread = threading.current_thread()
                original_name = current_thread.name
                current_thread.name = thread_name
                try:
                    result = slack_handler._is_background_thread()
                    results.append((thread_name, result, expected))
                finally:
                    current_thread.name = original_name

            if thread_name == "MainThread":
                # Test in current thread (should be MainThread)
                thread_task()
            else:
                # Test in background thread
                thread = threading.Thread(target=thread_task, name=thread_name)
                thread.start()
                thread.join()

        # Run all test cases
        for thread_name, expected in test_cases:
            test_thread_name(thread_name, expected)

        # Verify results
        for thread_name, actual, expected in results:
            assert (
                actual == expected
            ), f"Thread '{thread_name}' detection failed: expected {expected}, got {actual}"

    @pytest.mark.asyncio
    async def test_concurrent_notifications(self, slack_handler):
        """Test thread safety with concurrent notifications."""
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "ts": "1234567890.123456"}

        results = []

        async def send_notification(message_id):
            with patch("requests.post", return_value=mock_response):
                with patch("asyncio.get_event_loop") as mock_get_loop:
                    mock_loop = Mock()
                    mock_get_loop.return_value = mock_loop
                    mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

                    result = await slack_handler._send_via_api_threadsafe(
                        f"Message {message_id}",
                        "#test",
                        MessageFormat.PLAIN_TEXT,
                        [],
                        {},
                    )
                    results.append((message_id, result["success"]))

        def run_concurrent_test():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Send multiple notifications concurrently
                tasks = [send_notification(i) for i in range(5)]
                loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                loop.close()

        # Run in background thread
        thread = threading.Thread(
            target=run_concurrent_test, name="ConcurrentTestThread"
        )
        thread.start()
        thread.join()

        # Verify all notifications succeeded
        assert len(results) == 5
        for message_id, success in results:
            assert success is True, f"Message {message_id} failed"
