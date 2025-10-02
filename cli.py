#!/usr/bin/env python3
"""
StrandsAgent Universal Agent System CLI

Interactive command-line interface for the StrandsAgent Universal Agent System.
Provides workflow execution, system monitoring, and management capabilities.
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from supervisor.supervisor import Supervisor
from common.request_model import RequestMetadata

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cli")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="StrandsAgent Universal Agent System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                          # Interactive mode
  python cli.py --config custom.yaml    # Use custom config
  python cli.py --workflow "Search for weather in Seattle"  # Single workflow
  python cli.py --status                # Show system status
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--workflow", "-w",
        help="Execute a single workflow and exit"
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show system status and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
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
    try:
        logger.info("Starting system...")
        supervisor.start()
        
        logger.info(f"Executing workflow: {workflow_instruction}")
        workflow_id = supervisor.workflow_engine.start_workflow(workflow_instruction)
        logger.info(f"✅ Workflow '{workflow_id}' started")
        
        # Monitor workflow progress
        workflow_completed = False
        while not workflow_completed:
            progress_info = supervisor.workflow_engine.get_request_status(workflow_id)
            if progress_info is None:
                workflow_completed = True
                logger.info("✅ Workflow completed")
            else:
                logger.info(f"Workflow Status: {progress_info}")
                # Fixed: Check "is_completed" instead of "status"
                if progress_info.get("is_completed", False):
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
            if 'workflow_engine' in status and status['workflow_engine']:
                we_status = status['workflow_engine']
                print(f"\n📊 WorkflowEngine:")
                print(f"  State: {we_status.get('state', 'unknown')}")
                print(f"  Active Workflows: {we_status.get('active_workflows', 0)}")
                print(f"  Queue Size: {we_status.get('queue_size', 0)}")
            
            # Universal Agent status
            if 'universal_agent' in status and status['universal_agent']:
                ua_status = status['universal_agent']
                print(f"\n🤖 Universal Agent:")
                print(f"  Framework: {ua_status.get('framework', 'unknown')}")
                print(f"  Enabled: {ua_status.get('universal_agent_enabled', False)}")
                print(f"  Available Roles: {', '.join(ua_status.get('available_roles', []))}")
            
            # Heartbeat status
            if 'heartbeat' in status and status['heartbeat']:
                hb_status = status['heartbeat']
                print(f"\n💓 Heartbeat:")
                print(f"  Overall Status: {hb_status.get('overall_status', 'unknown')}")
                print(f"  Components Healthy: {hb_status.get('components_healthy', 0)}")
                
            print("=" * 50)
        else:
            logger.error("❌ Failed to retrieve system status")
            
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")


def run_interactive_mode(supervisor: Supervisor):
    """Run interactive CLI mode."""
    try:
        logger.info("Starting StrandsAgent Universal Agent System...")
        supervisor.start()
        logger.info("✅ System started successfully")
        
        print("\n🚀 StrandsAgent Universal Agent System - Interactive Mode")
        print("=" * 60)
        print("💬 Default: Enter any text to execute as a workflow")
        print("🔧 Slash commands:")
        print("  /status   - Show system status and metrics")
        print("  /stop     - Stop the system and exit")
        print("  /help     - Show this help message")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n➤ ").strip()
                
                # Handle slash commands
                if user_input.startswith("/"):
                    command = user_input[1:].lower()
                    
                    if command == "stop":
                        logger.info("Stopping system...")
                        supervisor.stop()
                        logger.info("✅ System stopped successfully")
                        break
                        
                    elif command == "status":
                        show_system_status(supervisor)
                        
                    elif command == "help":
                        print("\n🔧 Available Commands:")
                        print("  /status   - Show system status and metrics")
                        print("  /stop     - Stop the system and exit")
                        print("  /help     - Show this help message")
                        print("💬 Default: Any other text will be executed as a workflow")
                        
                    else:
                        print(f"❌ Unknown command: {user_input}")
                        print("💡 Type /help for available commands")
                        
                # Default: treat as workflow instruction
                elif user_input:
                    workflow_instruction = user_input
                    
                    if len(workflow_instruction) < 5:
                        print("⚠️ Workflow instruction too short. Please enter at least 5 characters.")
                        continue
                    
                    # Execute workflow
                    logger.info(f"🔄 Starting workflow: {workflow_instruction}")
                    workflow_id = supervisor.workflow_engine.start_workflow(workflow_instruction)
                    logger.info(f"✅ Workflow '{workflow_id}' started")
                    
                    # Monitor progress
                    workflow_completed = False
                    while not workflow_completed:
                        progress_info = supervisor.workflow_engine.get_request_status(workflow_id)
                        if progress_info is None:
                            workflow_completed = True
                            logger.info("✅ Workflow completed")
                        else:
                            logger.info(f"📊 Workflow Status: {progress_info}")
                            # Fixed: Check "is_completed" instead of "status"
                            if progress_info.get("is_completed", False):
                                workflow_completed = True
                                logger.info("✅ Workflow completed successfully")
                            else:
                                time.sleep(3)  # Check every 3 seconds
                        
                # Handle empty input
                else:
                    print("💡 Enter a workflow instruction or use /help for commands")
                    continue
                    
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
            supervisor.stop()
            logger.info("✅ System shutdown complete")
        except:
            pass


if __name__ == "__main__":
    main()