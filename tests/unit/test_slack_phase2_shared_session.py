"""Unit tests for Phase 2: Slack shared session pool functionality.

Tests the shared aiohttp session pool implementation for better performance
and connection reuse in the SlackHandler.
"""

import asyncio
import threading
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import MessageFormat


class TestSlackPhase2SharedSession:
    """Test suite for Phase 2 shared session pool implementation."""

    @pytest.fixture
    def slack_handler_with_shared_session(self):
        """Create SlackHandler instance with shared session enabled."""
        config = {
            "bot_token": "xoxb-test-token",
            "default_channel": "#test",
            "use_shared_session": True,
            "session_pool_size": 5,
        }
        return SlackChannelHandler(config)

    @pytest.fixture
    def slack_handler_without_shared_session(self):
        """Create SlackHandler instance with shared session disabled."""
        config = {
            "bot_token": "xoxb-test-token",
            "default_channel": "#test",
            "use_shared_session": False,
            "session_pool_size": 5,
        }
        return SlackChannelHandler(config)

    @pytest.mark.asyncio
    async def test_shared_session_creation(self, slack_handler_with_shared_session):
        """Test that shared session is created with proper configuration."""
        handler = slack_handler_with_shared_session

        with patch("aiohttp.ClientSession") as mock_session_class:
            with patch("aiohttp.TCPConnector") as mock_connector_class:
                mock_session = Mock()
                mock_session.closed = False
                mock_session_class.return_value = mock_session

                mock_connector = Mock()
                mock_connector_class.return_value = mock_connector

                session = await handler._get_or_create_session()

                # Verify session creation
                assert session == mock_session
                mock_connector_class.assert_called_once_with(
                    limit=5,
                    limit_per_host=5,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True,
                )
                mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_shared_session_reuse(self, slack_handler_with_shared_session):
        """Test that shared session is reused across multiple calls."""
        handler = slack_handler_with_shared_session

        with patch("aiohttp.ClientSession") as mock_session_class:
            with patch("aiohttp.TCPConnector"):
                mock_session = Mock()
                mock_session.closed = False
                mock_session_class.return_value = mock_session

                # First call should create session
                session1 = await handler._get_or_create_session()

                # Second call should reuse same session
                session2 = await handler._get_or_create_session()

                assert session1 == session2
                # Session should only be created once
                mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_shared_session_recreation_when_closed(
        self, slack_handler_with_shared_session
    ):
        """Test that shared session is recreated when previous session is closed."""
        handler = slack_handler_with_shared_session

        with patch("aiohttp.ClientSession") as mock_session_class:
            with patch("aiohttp.TCPConnector"):
                # First session (will be marked as closed)
                mock_session1 = Mock()
                mock_session1.closed = False

                # Second session (new one)
                mock_session2 = Mock()
                mock_session2.closed = False

                mock_session_class.side_effect = [mock_session1, mock_session2]

                # First call creates session
                session1 = await handler._get_or_create_session()
                assert session1 == mock_session1

                # Mark first session as closed
                mock_session1.closed = True

                # Second call should create new session
                session2 = await handler._get_or_create_session()
                assert session2 == mock_session2

                # Should have created two sessions
                assert mock_session_class.call_count == 2

    @pytest.mark.asyncio
    async def test_shared_session_thread_safety(
        self, slack_handler_with_shared_session
    ):
        """Test that shared session creation is thread-safe."""
        handler = slack_handler_with_shared_session

        with patch("aiohttp.ClientSession") as mock_session_class:
            with patch("aiohttp.TCPConnector"):
                mock_session = Mock()
                mock_session.closed = False
                mock_session_class.return_value = mock_session

                # Simulate concurrent access
                sessions = []

                async def get_session():
                    session = await handler._get_or_create_session()
                    sessions.append(session)

                # Run multiple concurrent calls
                await asyncio.gather(*[get_session() for _ in range(5)])

                # All should get the same session
                assert all(session == mock_session for session in sessions)
                # Session should only be created once despite concurrent access
                mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_thread_uses_shared_session_when_enabled(
        self, slack_handler_with_shared_session
    ):
        """Test that main thread uses shared session when enabled."""
        handler = slack_handler_with_shared_session

        with patch.object(
            handler, "_send_via_shared_session", new_callable=AsyncMock
        ) as mock_shared:
            with patch.object(
                handler, "_send_via_api_aiohttp", new_callable=AsyncMock
            ) as mock_aiohttp:
                with patch.object(
                    handler, "_send_via_api_threadsafe", new_callable=AsyncMock
                ) as mock_threadsafe:
                    mock_shared.return_value = {"success": True}

                    await handler._send_via_api(
                        "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
                    )

                    # Should use shared session, not standard aiohttp or threadsafe
                    mock_shared.assert_called_once()
                    mock_aiohttp.assert_not_called()
                    mock_threadsafe.assert_not_called()

    @pytest.mark.asyncio
    async def test_main_thread_uses_standard_aiohttp_when_disabled(
        self, slack_handler_without_shared_session
    ):
        """Test that main thread uses standard aiohttp when shared session is disabled."""
        handler = slack_handler_without_shared_session

        with patch.object(
            handler, "_send_via_shared_session", new_callable=AsyncMock
        ) as mock_shared:
            with patch.object(
                handler, "_send_via_api_aiohttp", new_callable=AsyncMock
            ) as mock_aiohttp:
                with patch.object(
                    handler, "_send_via_api_threadsafe", new_callable=AsyncMock
                ) as mock_threadsafe:
                    mock_aiohttp.return_value = {"success": True}

                    await handler._send_via_api(
                        "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
                    )

                    # Should use standard aiohttp, not shared session or threadsafe
                    mock_aiohttp.assert_called_once()
                    mock_shared.assert_not_called()
                    mock_threadsafe.assert_not_called()

    @pytest.mark.asyncio
    async def test_shared_session_http_success(self, slack_handler_with_shared_session):
        """Test shared session HTTP client with successful response."""
        handler = slack_handler_with_shared_session

        # Mock successful Slack API response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "ts": "1234567890.123456", "channel": "#test"}
        )

        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.post.return_value.__aexit__.return_value = None

        with patch.object(handler, "_get_or_create_session", return_value=mock_session):
            result = await handler._send_via_shared_session(
                "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
            )

            # Verify success response
            assert result["success"] is True
            assert result["channel"] == "#test"
            assert result["ts"] == "1234567890.123456"
            assert result["message_id"] == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_shared_session_http_api_error(
        self, slack_handler_with_shared_session
    ):
        """Test shared session HTTP client with Slack API error."""
        handler = slack_handler_with_shared_session

        # Mock Slack API error response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": False, "error": "channel_not_found"}
        )

        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.post.return_value.__aexit__.return_value = None

        with patch.object(handler, "_get_or_create_session", return_value=mock_session):
            result = await handler._send_via_shared_session(
                "test message", "#invalid", MessageFormat.PLAIN_TEXT, [], {}
            )

            # Verify error response
            assert result["success"] is False
            assert "channel_not_found" in result["error"]

    @pytest.mark.asyncio
    async def test_shared_session_http_error(self, slack_handler_with_shared_session):
        """Test shared session HTTP client with HTTP error."""
        handler = slack_handler_with_shared_session

        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")

        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.post.return_value.__aexit__.return_value = None

        with patch.object(handler, "_get_or_create_session", return_value=mock_session):
            result = await handler._send_via_shared_session(
                "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
            )

            # Verify error response
            assert result["success"] is False
            assert "404" in result["error"]
            assert "Not Found" in result["error"]

    @pytest.mark.asyncio
    async def test_shared_session_timeout_error(
        self, slack_handler_with_shared_session
    ):
        """Test shared session HTTP client with timeout error."""
        handler = slack_handler_with_shared_session

        mock_session = Mock()
        mock_session.post.side_effect = asyncio.TimeoutError("Request timed out")

        with patch.object(handler, "_get_or_create_session", return_value=mock_session):
            result = await handler._send_via_shared_session(
                "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
            )

            # Verify timeout error response
            assert result["success"] is False
            assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_shared_session_exception_handling(
        self, slack_handler_with_shared_session
    ):
        """Test shared session exception handling and session reset."""
        handler = slack_handler_with_shared_session

        mock_session = Mock()
        mock_session.closed = False
        mock_session.post.side_effect = Exception("Connection error")
        mock_session.close = AsyncMock()

        with patch.object(handler, "_get_or_create_session", return_value=mock_session):
            result = await handler._send_via_shared_session(
                "test message", "#test", MessageFormat.PLAIN_TEXT, [], {}
            )

            # Verify error response
            assert result["success"] is False
            assert "Connection error" in result["error"]

            # Verify session was closed and reset
            mock_session.close.assert_called_once()
            assert handler._session is None

    @pytest.mark.asyncio
    async def test_shared_session_with_buttons(self, slack_handler_with_shared_session):
        """Test shared session with buttons metadata."""
        handler = slack_handler_with_shared_session

        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "ts": "1234567890.123456"}
        )

        mock_session = Mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session.post.return_value.__aexit__.return_value = None

        buttons = [
            {"text": "Yes", "value": "yes", "style": "primary"},
            {"text": "No", "value": "no", "style": "default"},
        ]

        with patch.object(handler, "_get_or_create_session", return_value=mock_session):
            result = await handler._send_via_shared_session(
                "test message", "#test", MessageFormat.RICH_TEXT, buttons, {}
            )

            # Verify success and that session was used
            assert result["success"] is True
            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_shared_session_cleanup(self, slack_handler_with_shared_session):
        """Test shared session cleanup functionality."""
        handler = slack_handler_with_shared_session

        # Create a mock session
        mock_session = Mock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        # Set the session
        handler._session = mock_session

        # Call cleanup
        await handler._cleanup_shared_session()

        # Verify session was closed and references cleared
        mock_session.close.assert_called_once()
        assert handler._session is None
        assert handler._session_lock is None

    @pytest.mark.asyncio
    async def test_shared_session_cleanup_already_closed(
        self, slack_handler_with_shared_session
    ):
        """Test shared session cleanup when session is already closed."""
        handler = slack_handler_with_shared_session

        # Create a mock session that's already closed
        mock_session = Mock()
        mock_session.closed = True
        mock_session.close = AsyncMock()

        # Set the session
        handler._session = mock_session

        # Call cleanup
        await handler._cleanup_shared_session()

        # Verify close was not called since session was already closed
        mock_session.close.assert_not_called()
        assert handler._session is None

    @pytest.mark.asyncio
    async def test_shared_session_cleanup_exception(
        self, slack_handler_with_shared_session
    ):
        """Test shared session cleanup handles exceptions gracefully."""
        handler = slack_handler_with_shared_session

        # Create a mock session that throws exception on close
        mock_session = Mock()
        mock_session.closed = False
        mock_session.close = AsyncMock(side_effect=Exception("Close error"))

        # Set the session
        handler._session = mock_session

        # Call cleanup (should not raise exception)
        await handler._cleanup_shared_session()

        # Verify references were still cleared despite exception
        assert handler._session is None
        assert handler._session_lock is None

    @pytest.mark.asyncio
    async def test_stop_session_calls_cleanup(self, slack_handler_with_shared_session):
        """Test that stop_session calls shared session cleanup."""
        handler = slack_handler_with_shared_session

        with patch.object(
            handler, "_cleanup_shared_session", new_callable=AsyncMock
        ) as mock_cleanup:
            await handler.stop_session()

            # Verify cleanup was called
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_comparison_simulation(
        self, slack_handler_with_shared_session
    ):
        """Test performance characteristics of shared session vs individual sessions."""
        handler = slack_handler_with_shared_session

        # Mock session creation timing
        session_creation_times = []

        def mock_session_creation(*args, **kwargs):
            session_creation_times.append(time.time())
            mock_session = Mock()
            mock_session.closed = False
            return mock_session

        with patch("aiohttp.ClientSession", side_effect=mock_session_creation):
            with patch("aiohttp.TCPConnector"):
                # Multiple calls should reuse session
                for _ in range(5):
                    await handler._get_or_create_session()

        # Should only create session once
        assert len(session_creation_times) == 1
