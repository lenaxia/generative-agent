"""
Tests for Slack session cleanup and shutdown functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.channel_handlers.slack_handler import SlackChannelHandler


class TestSlackSessionCleanup:
    """Test cases for Slack session cleanup and shutdown."""

    @pytest.fixture
    def slack_handler(self):
        """Create a SlackChannelHandler instance for testing."""
        config = {
            "bot_token": "xoxb-test-token",
            "app_token": "xapp-test-token",
            "default_channel": "#general",
        }
        return SlackChannelHandler(config)

    @pytest.mark.asyncio
    async def test_stop_session_sets_shutdown_flag(self, slack_handler):
        """Test that stop_session sets the shutdown flag."""
        assert slack_handler.shutdown_flag is False

        await slack_handler.stop_session()

        assert slack_handler.shutdown_flag is True

    @pytest.mark.asyncio
    async def test_stop_session_clears_pending_questions(self, slack_handler):
        """Test that stop_session clears pending questions."""
        # Add some pending questions
        slack_handler.pending_questions = {
            "question_1": {"question": "test", "response_future": asyncio.Future()},
            "question_2": {"question": "test2", "response_future": asyncio.Future()},
        }

        await slack_handler.stop_session()

        assert len(slack_handler.pending_questions) == 0

    @pytest.mark.asyncio
    async def test_stop_session_marks_session_inactive(self, slack_handler):
        """Test that stop_session marks session as inactive."""
        slack_handler.session_active = True

        await slack_handler.stop_session()

        assert slack_handler.session_active is False

    @pytest.mark.asyncio
    async def test_stop_session_cleans_up_socket_handler_with_close(
        self, slack_handler
    ):
        """Test that stop_session calls close on socket handler if available."""
        mock_socket_handler = MagicMock()
        mock_socket_handler.close = MagicMock()
        slack_handler.socket_handler = mock_socket_handler

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock()

            await slack_handler.stop_session()

            mock_loop.run_in_executor.assert_called_once_with(
                None, mock_socket_handler.close
            )

    @pytest.mark.asyncio
    async def test_stop_session_cleans_up_socket_handler_with_stop(self, slack_handler):
        """Test that stop_session calls stop on socket handler if close not available."""
        mock_socket_handler = MagicMock()
        mock_socket_handler.stop = MagicMock()
        # Don't add close method
        slack_handler.socket_handler = mock_socket_handler

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock()

            await slack_handler.stop_session()

            mock_loop.run_in_executor.assert_called_once_with(
                None, mock_socket_handler.stop
            )

    @pytest.mark.asyncio
    async def test_stop_session_handles_cleanup_errors_gracefully(self, slack_handler):
        """Test that stop_session handles cleanup errors gracefully."""
        mock_socket_handler = MagicMock()
        mock_socket_handler.close = MagicMock(side_effect=Exception("Cleanup error"))
        slack_handler.socket_handler = mock_socket_handler

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock(
                side_effect=Exception("Cleanup error")
            )

            # Should not raise exception
            await slack_handler.stop_session()

            # Should still cleanup references
            assert slack_handler.slack_app is None
            assert slack_handler.socket_handler is None
            assert slack_handler.shutdown_flag is True

    @pytest.mark.asyncio
    async def test_stop_session_cleans_up_references(self, slack_handler):
        """Test that stop_session cleans up slack_app and socket_handler references."""
        slack_handler.slack_app = MagicMock()
        slack_handler.socket_handler = MagicMock()

        await slack_handler.stop_session()

        assert slack_handler.slack_app is None
        assert slack_handler.socket_handler is None

    @pytest.mark.asyncio
    async def test_run_socket_handler_with_shutdown_handles_shutdown_flag(
        self, slack_handler
    ):
        """Test that _run_socket_handler_with_shutdown respects shutdown flag."""
        mock_socket_handler = MagicMock()
        mock_socket_handler.start_async = AsyncMock(
            side_effect=Exception("Connection error")
        )
        slack_handler.socket_handler = mock_socket_handler
        slack_handler.shutdown_flag = True

        # Should not raise exception when shutdown flag is set
        await slack_handler._run_socket_handler_with_shutdown()

    @pytest.mark.asyncio
    async def test_run_socket_handler_with_shutdown_raises_on_error_when_not_shutting_down(
        self, slack_handler
    ):
        """Test that _run_socket_handler_with_shutdown raises errors when not shutting down."""
        mock_socket_handler = MagicMock()
        mock_socket_handler.start_async = AsyncMock(
            side_effect=Exception("Connection error")
        )
        slack_handler.socket_handler = mock_socket_handler
        slack_handler.shutdown_flag = False

        with pytest.raises(Exception, match="Connection error"):
            await slack_handler._run_socket_handler_with_shutdown()

    @pytest.mark.asyncio
    async def test_run_socket_handler_with_shutdown_tries_async_first(
        self, slack_handler
    ):
        """Test that _run_socket_handler_with_shutdown tries start_async first."""
        mock_socket_handler = MagicMock()
        mock_socket_handler.start_async = AsyncMock()
        slack_handler.socket_handler = mock_socket_handler

        await slack_handler._run_socket_handler_with_shutdown()

        mock_socket_handler.start_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_socket_handler_with_shutdown_falls_back_to_sync(
        self, slack_handler
    ):
        """Test that _run_socket_handler_with_shutdown falls back to sync start."""
        mock_socket_handler = MagicMock()
        # Don't add start_async method
        mock_socket_handler.start = MagicMock()
        slack_handler.socket_handler = mock_socket_handler

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock()

            await slack_handler._run_socket_handler_with_shutdown()

            mock_loop.run_in_executor.assert_called_once_with(
                None, mock_socket_handler.start
            )

    def test_shutdown_flag_initialized_to_false(self, slack_handler):
        """Test that shutdown_flag is initialized to False."""
        assert slack_handler.shutdown_flag is False
