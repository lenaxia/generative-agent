#!/usr/bin/env python3
"""Universal Agent System CLI.

Interactive command-line interface for the Universal Agent System.
Provides workflow execution, system monitoring, and management capabilities.
"""

import argparse
import logging
import os
import readline
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from supervisor.supervisor import Supervisor  # noqa: E402
from supervisor.workflow_duration_logger import WorkflowSource  # noqa: E402

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cli")


def setup_readline():
    """Setup readline for command history and cursor navigation."""
    try:
        # Set up history file
        history_file = os.path.expanduser("~/.generative_agent_history")

        # Load existing history
        if os.path.exists(history_file):
            readline.read_history_file(history_file)

        # Set history length
        readline.set_history_length(1000)

        # Enable tab completion (basic)
        readline.parse_and_bind("tab: complete")

        # Enable emacs-style key bindings (default)
        readline.parse_and_bind("set editing-mode emacs")

        # Save history on exit
        import atexit

        atexit.register(lambda: readline.write_history_file(history_file))

        logger.debug("Readline setup completed with history file: %s", history_file)

    except Exception as e:
        logger.warning("Could not setup readline: %s", e)


def show_command_history():
    """Show the command history."""
    try:
        history_length = readline.get_current_history_length()
        if history_length == 0:
            print("📝 No command history available")
            return

        print(f"\n📝 Command History (last {min(history_length, 20)} commands):")
        print("-" * 50)

        # Show last 20 commands
        start_idx = max(1, history_length - 19)
        for i in range(start_idx, history_length + 1):
            try:
                cmd = readline.get_history_item(i)
                if cmd:
                    print(f"{i:3d}: {cmd}")
            except (IndexError, OSError):
                continue
        print("-" * 50)

    except Exception as e:
        logger.warning("Could not show command history: %s", e)
        print("❌ Could not retrieve command history")


def clear_command_history():
    """Clear the command history."""
    try:
        readline.clear_history()
        logger.debug("Command history cleared")
    except Exception as e:
        logger.warning("Could not clear command history: %s", e)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Universal Agent System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                          # Interactive mode
  python cli.py --config custom.yaml    # Use custom config
  python cli.py --workflow "Search for weather in Seattle"  # Single workflow
  python cli.py --status                # Show system status
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )

    parser.add_argument("--workflow", "-w", help="Execute a single workflow and exit")

    parser.add_argument(
        "--status", "-s", action="store_true", help="Show system status and exit"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize Supervisor
    try:
        logger.info("Initializing StrandsAgent Universal Agent System...")
        supervisor = Supervisor(args.config)
        logger.info("✅ System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        sys.exit(1)

    # Handle single workflow execution
    if args.workflow:
        execute_single_workflow(supervisor, args.workflow)
        return

    # Handle status request
    if args.status:
        show_system_status(supervisor)
        return

    # Interactive mode
    run_interactive_mode(supervisor)


def execute_single_workflow(supervisor: Supervisor, workflow_instruction: str):
    """Execute a single workflow and exit."""
    workflow_id = None

    try:
        logger.info("Starting system...")
        supervisor.start()

        logger.info(f"Executing workflow: {workflow_instruction}")
        workflow_id = supervisor.workflow_engine.start_workflow(workflow_instruction)
        logger.info(f"✅ Workflow '{workflow_id}' started")

        # Update workflow source to CLI (duration tracking is handled in workflow engine)
        supervisor.workflow_engine.update_workflow_source(
            workflow_id, WorkflowSource.CLI
        )

        # Monitor workflow progress
        workflow_completed = False

        while not workflow_completed:
            progress_info = supervisor.workflow_engine.get_request_status(workflow_id)
            if progress_info is None:
                workflow_completed = True
                logger.info("✅ Workflow completed")
            else:
                logger.info(f"Workflow Status: {progress_info}")

                # Check for errors
                if progress_info.get("error"):
                    workflow_completed = True
                    logger.error(f"❌ Workflow failed: {progress_info.get('error')}")
                # Check "is_completed" instead of "status"
                elif progress_info.get("is_completed", False):
                    workflow_completed = True
                    logger.info("✅ Workflow completed successfully")
                else:
                    time.sleep(2)  # Check every 2 seconds

    except KeyboardInterrupt:
        logger.info("⚠️ Workflow interrupted by user")
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
    finally:
        logger.info("Stopping system...")
        supervisor.stop()


def show_system_status(supervisor: Supervisor):
    """Show system status and exit."""
    try:
        status = supervisor.status()
        if status:
            print("\n🔍 StrandsAgent Universal Agent System Status:")
            print("=" * 50)
            print(f"System Status: {status.get('status', 'unknown')}")
            print(f"Running: {status.get('running', False)}")

            # WorkflowEngine status
            if "workflow_engine" in status and status["workflow_engine"]:
                we_status = status["workflow_engine"]
                print("\n📊 WorkflowEngine:")
                print(f"  State: {we_status.get('state', 'unknown')}")
                print(f"  Active Workflows: {we_status.get('active_workflows', 0)}")
                print(f"  Queue Size: {we_status.get('queue_size', 0)}")

            # Universal Agent status
            if "universal_agent" in status and status["universal_agent"]:
                ua_status = status["universal_agent"]
                print("\n🤖 Universal Agent:")
                print(f"  Framework: {ua_status.get('framework', 'unknown')}")
                print(f"  Enabled: {ua_status.get('universal_agent_enabled', False)}")
                print(
                    f"  Available Roles: {', '.join(ua_status.get('available_roles', []))}"
                )

            # Heartbeat status
            if "heartbeat" in status and status["heartbeat"]:
                hb_status = status["heartbeat"]
                print("\n💓 Heartbeat:")
                print(f"  Overall Status: {hb_status.get('overall_status', 'unknown')}")
                print(f"  Components Healthy: {hb_status.get('components_healthy', 0)}")

            print("=" * 50)
        else:
            logger.error("❌ Failed to retrieve system status")

    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")


def _print_welcome_message():
    """Print the welcome message and help information."""
    print("\n🚀 StrandsAgent Universal Agent System - Interactive Mode")
    print("=" * 60)
    print("💬 Default: Enter any text to execute as a workflow")
    print("🔧 Slash commands:")
    print("  /status   - Show system status and metrics")
    print("  /exit     - Exit the system and exit")
    print("  /help     - Show this help message")
    print("  /history  - Show command history")
    print("  /clear    - Clear command history")
    print("📝 Navigation:")
    print("  ↑/↓ arrows - Navigate command history")
    print("  ←/→ arrows - Move cursor within current line")
    print("  Ctrl+A/E  - Move to beginning/end of line")
    print("=" * 60)


def _print_help_message():
    """Print the help message with available commands."""
    print("\n🔧 Available Commands:")
    print("  /status   - Show system status and metrics")
    print("  /exit     - Stop the system and exit")
    print("  /help     - Show this help message")
    print("  /history  - Show command history")
    print("  /clear    - Clear command history")
    print("📝 Navigation:")
    print("  ↑/↓ arrows - Navigate command history")
    print("  ←/→ arrows - Move cursor within current line")
    print("  Ctrl+A/E  - Move to beginning/end of line")
    print("💬 Default: Any other text will be executed as a workflow")


def _handle_slash_command(command: str, supervisor: Supervisor) -> bool:
    """Handle slash commands.

    Args:
        command: The command string (without the leading slash)
        supervisor: The supervisor instance

    Returns:
        True if the command was 'exit', False otherwise
    """
    if command == "exit":
        logger.info("Stopping system...")
        supervisor.stop()
        logger.info("✅ System stopped successfully")
        return True
    elif command == "status":
        show_system_status(supervisor)
    elif command == "help":
        _print_help_message()
    elif command == "history":
        show_command_history()
    elif command == "clear":
        clear_command_history()
        print("✅ Command history cleared")
    else:
        print(f"❌ Unknown command: /{command}")
        print("💡 Type /help for available commands")

    return False


def _execute_workflow(workflow_instruction: str, supervisor: Supervisor):
    """Execute a workflow instruction.

    Args:
        workflow_instruction: The workflow instruction to execute
        supervisor: The supervisor instance
    """
    if len(workflow_instruction) < 5:
        print("⚠️ Workflow instruction too short. Please enter at least 5 characters.")
        return

    # Execute workflow
    logger.info(f"🔄 Starting workflow: {workflow_instruction}")
    workflow_id = supervisor.workflow_engine.start_workflow(workflow_instruction)

    # Update workflow source to CLI
    supervisor.workflow_engine.update_workflow_source(workflow_id, WorkflowSource.CLI)

    # Monitor progress
    _monitor_workflow_progress(workflow_id, supervisor)


def _monitor_workflow_progress(workflow_id: str, supervisor: Supervisor):
    """Monitor workflow progress until completion.

    Args:
        workflow_id: The ID of the workflow to monitor
        supervisor: The supervisor instance
    """
    workflow_completed = False

    try:
        while not workflow_completed:
            progress_info = supervisor.workflow_engine.get_request_status(workflow_id)
            if progress_info is None:
                workflow_completed = True
                logger.info("✅ Workflow completed")
            else:
                logger.info(f"📊 Workflow Status: {progress_info}")

                # Check for errors
                if progress_info.get("error"):
                    workflow_completed = True
                    logger.error(f"❌ Workflow failed: {progress_info.get('error')}")
                # Check "is_completed" instead of "status"
                elif progress_info.get("is_completed", False):
                    workflow_completed = True
                    logger.info("✅ Workflow completed successfully")
                else:
                    time.sleep(3)  # Check every 3 seconds

    except KeyboardInterrupt:
        logger.info("⚠️ Workflow interrupted by user")


def _process_user_input(user_input: str, supervisor: Supervisor) -> bool:
    """Process user input and execute appropriate action.

    Args:
        user_input: The user's input string
        supervisor: The supervisor instance

    Returns:
        True if the system should exit, False otherwise
    """
    # Handle slash commands
    if user_input.startswith("/"):
        command = user_input[1:].lower()
        return _handle_slash_command(command, supervisor)

    # Default: treat as workflow instruction
    elif user_input:
        _execute_workflow(user_input, supervisor)

    # Handle empty input
    else:
        print("💡 Enter a workflow instruction or use /help for commands")

    return False


def run_interactive_mode(supervisor: Supervisor):
    """Run interactive CLI mode."""
    try:
        logger.info("Starting StrandsAgent Universal Agent System...")
        supervisor.start()
        logger.info("✅ System started successfully")

        # Setup readline for command history and cursor navigation
        setup_readline()
        _print_welcome_message()

        while True:
            try:
                user_input = input("\n➤ ").strip()
                should_exit = _process_user_input(user_input, supervisor)
                if should_exit:
                    break

            except KeyboardInterrupt:
                logger.info("\n⚠️ Interrupted by user")
                continue

    except KeyboardInterrupt:
        logger.info("\n⚠️ System interrupted by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        sys.exit(1)
    finally:
        try:
            # Clean up readline before stopping supervisor
            try:
                import atexit
                import readline

                # Save history before cleanup
                history_file = os.path.expanduser("~/.strandsagent_history")
                if os.path.exists(history_file):
                    readline.write_history_file(history_file)

                # Clear readline to help with thread cleanup
                readline.clear_history()

            except Exception as e:
                logger.debug(f"Readline cleanup error: {e}")

            supervisor.stop()
            logger.info("✅ System shutdown complete")

            # Force exit to prevent hanging on slack-bolt internal threads
            # This is necessary because slack-bolt's SocketModeHandler creates
            # internal threads that don't terminate cleanly during Python shutdown.
            # Our application components shut down properly, but the third-party
            # library threads cause hanging in threading._shutdown().
            import os
            import time

            time.sleep(0.1)  # Brief pause for final cleanup
            logger.info("🔄 Force exiting to prevent slack-bolt thread hanging...")
            os._exit(0)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            import os

            os._exit(1)


if __name__ == "__main__":
    main()
