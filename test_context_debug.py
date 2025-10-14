#!/usr/bin/env python3
"""Test script to examine Strands context injection in timer role."""

import asyncio
import logging
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging to see our debug output
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from supervisor.supervisor import Supervisor


async def test_context_debug():
    """Test the debug_context tool to see what's available."""
    print("🔍 Testing Strands context injection...")

    try:
        # Initialize supervisor
        supervisor = Supervisor("config.yaml")
        supervisor.start()

        print("✅ Supervisor started")

        # Test the debug context tool
        print("📞 Calling debug_context tool via timer role...")

        # Use the workflow engine to execute a timer role task that calls debug_context
        workflow_id = supervisor.workflow_engine.start_workflow(
            "Use the debug_context tool to show me what context is available",
            source_id="test_script",
            target_id="timer",
        )

        print(f"🚀 Started workflow: {workflow_id}")

        # Wait a moment for execution
        await asyncio.sleep(2)

        # Check workflow status
        if workflow_id in supervisor.workflow_engine.active_workflows:
            context = supervisor.workflow_engine.active_workflows[workflow_id]
            print(f"📊 Workflow status: {context.execution_state}")

        print("✅ Test completed - check logs for debug output")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        try:
            supervisor.stop()
            print("🛑 Supervisor stopped")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_context_debug())
