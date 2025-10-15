"""Test CLI complexity refactor to ensure functionality is preserved."""

import io
from unittest.mock import Mock, patch

import pytest

from cli import run_interactive_mode
from supervisor.supervisor import Supervisor


class TestCLIComplexityRefactor:
    """Test CLI refactoring maintains functionality."""

    @pytest.fixture
    def mock_supervisor(self):
        """Create a mock supervisor for testing."""
        supervisor = Mock(spec=Supervisor)
        supervisor.start = Mock()
        supervisor.stop = Mock()
        supervisor.workflow_engine = Mock()
        supervisor.workflow_engine.start_workflow = Mock(
            return_value="test-workflow-id"
        )
        supervisor.workflow_engine.update_workflow_source = Mock()
        supervisor.workflow_engine.get_request_status = Mock(return_value=None)
        return supervisor

    @patch("cli.setup_readline")
    @patch("cli._process_user_input")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_exit_command_functionality(
        self,
        mock_os_exit,
        mock_input,
        mock_process_input,
        mock_setup_readline,
        mock_supervisor,
    ):
        """Test that /exit command works correctly."""
        mock_input.return_value = "/exit"
        mock_process_input.return_value = True

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        mock_supervisor.start.assert_called_once()
        mock_process_input.assert_called_once_with("/exit", mock_supervisor)

    @patch("cli.setup_readline")
    @patch("cli.show_system_status")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_status_command_functionality(
        self,
        mock_os_exit,
        mock_input,
        mock_show_status,
        mock_setup_readline,
        mock_supervisor,
    ):
        """Test that /status command works correctly."""
        mock_input.side_effect = ["/status", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        mock_show_status.assert_called_once_with(mock_supervisor)

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_help_command_functionality(
        self, mock_os_exit, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test that /help command works correctly."""
        mock_input.side_effect = ["/help", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            await run_interactive_mode(mock_supervisor)

        output = mock_stdout.getvalue()
        assert "Available Commands:" in output

    @patch("cli.setup_readline")
    @patch("cli.show_command_history")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_history_command_functionality(
        self,
        mock_os_exit,
        mock_input,
        mock_show_history,
        mock_setup_readline,
        mock_supervisor,
    ):
        """Test that /history command works correctly."""
        mock_input.side_effect = ["/history", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        mock_show_history.assert_called_once()

    @patch("cli.setup_readline")
    @patch("cli.clear_command_history")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_clear_command_functionality(
        self,
        mock_os_exit,
        mock_input,
        mock_clear_history,
        mock_setup_readline,
        mock_supervisor,
    ):
        """Test that /clear command works correctly."""
        mock_input.side_effect = ["/clear", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        mock_clear_history.assert_called_once()

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_workflow_execution_functionality(
        self, mock_os_exit, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test that workflow execution works correctly."""
        mock_input.side_effect = ["Set a timer for 5 minutes", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        mock_supervisor.workflow_engine.start_workflow.assert_called_once()

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_short_workflow_instruction_handling(
        self, mock_os_exit, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test handling of short workflow instructions."""
        mock_input.side_effect = ["hi", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            await run_interactive_mode(mock_supervisor)

        output = mock_stdout.getvalue()
        assert "too short" in output

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_empty_input_handling(
        self, mock_os_exit, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test handling of empty input."""
        mock_input.side_effect = ["", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            await run_interactive_mode(mock_supervisor)

        output = mock_stdout.getvalue()
        assert "Enter a workflow instruction" in output

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_unknown_command_handling(
        self, mock_os_exit, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test handling of unknown commands."""
        mock_input.side_effect = ["/unknown", "/exit"]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            await run_interactive_mode(mock_supervisor)

        output = mock_stdout.getvalue()
        assert "Unknown command" in output

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_workflow_monitoring_functionality(
        self, mock_os_exit, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test workflow monitoring functionality."""
        mock_input.side_effect = ["Set a timer", "/exit"]

        mock_supervisor.workflow_engine.get_request_status.side_effect = [
            {"status": "running"},
            {"status": "completed"},
            None,
        ]

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        assert mock_supervisor.workflow_engine.get_request_status.call_count >= 1

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @patch("os._exit")
    @pytest.mark.asyncio
    async def test_workflow_error_handling(
        self, mock_os_exit, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test workflow error handling."""
        mock_input.side_effect = ["Set a timer", "/exit"]
        mock_supervisor.workflow_engine.start_workflow.side_effect = Exception(
            "Test error"
        )

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        mock_supervisor.workflow_engine.start_workflow.assert_called_once()

    @patch("cli.setup_readline")
    @patch("builtins.input")
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_handling(
        self, mock_input, mock_setup_readline, mock_supervisor
    ):
        """Test keyboard interrupt handling."""
        mock_input.side_effect = KeyboardInterrupt()

        with patch("sys.stdout", new_callable=io.StringIO):
            await run_interactive_mode(mock_supervisor)

        mock_supervisor.stop.assert_called()
