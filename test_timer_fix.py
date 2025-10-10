#!/usr/bin/env python3
"""Test script to verify timer functionality and identify issues."""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables to bypass shell prompts
os.environ["STRANDS_NON_INTERACTIVE"] = "true"
os.environ["BYPASS_TOOL_CONSENT"] = "true"


async def test_timer_tools():
    """Test timer tools directly."""
    print("=== Testing Timer Tools Directly ===")

    try:
        from roles.timer.tools import timer_cancel, timer_list, timer_set

        # Test timer_set
        print("\n1. Testing timer_set...")
        result = timer_set(
            duration="30s",
            label="Test Timer",
            user_id="test_user",
            channel_id="test_channel",
        )
        print(f"timer_set result: {result}")

        if result.get("success"):
            timer_id = result.get("timer_id")
            print(f"✅ Timer created successfully: {timer_id}")

            # Test timer_list
            print("\n2. Testing timer_list...")
            list_result = timer_list(user_id="test_user")
            print(f"timer_list result: {list_result}")

            if list_result.get("success"):
                timers = list_result.get("timers", [])
                print(f"✅ Found {len(timers)} timers")
                for timer in timers:
                    print(
                        f"   - {timer['timer_id']}: {timer['name']} ({timer['status']})"
                    )
            else:
                print(f"❌ timer_list failed: {list_result.get('error')}")

            # Test timer_cancel
            print("\n3. Testing timer_cancel...")
            cancel_result = timer_cancel(timer_id)
            print(f"timer_cancel result: {cancel_result}")

            if cancel_result.get("success"):
                print(f"✅ Timer cancelled successfully")
            else:
                print(f"❌ timer_cancel failed: {cancel_result.get('error')}")
        else:
            print(f"❌ timer_set failed: {result.get('error')}")

    except Exception as e:
        print(f"❌ Error testing timer tools: {e}")
        import traceback

        traceback.print_exc()


async def test_timer_lifecycle():
    """Test timer lifecycle functions."""
    print("\n=== Testing Timer Lifecycle ===")

    try:
        from roles.timer.lifecycle import (
            get_timer_manager,
            parse_timer_parameters,
            validate_timer_request,
        )

        # Test parameter parsing
        print("\n1. Testing parse_timer_parameters...")
        params = {"action": "set", "duration": "5m", "label": "Test Timer"}

        parsed = await parse_timer_parameters(params)
        print(f"Parsed parameters: {parsed}")

        # Test validation
        print("\n2. Testing validate_timer_request...")
        validation = await validate_timer_request(parsed)
        print(f"Validation result: {validation}")

        # Test timer manager directly
        print("\n3. Testing TimerManager...")
        timer_manager = get_timer_manager()

        timer_id = await timer_manager.create_timer(
            timer_type="countdown",
            duration_seconds=300,  # 5 minutes
            name="Direct Test Timer",
            user_id="test_user",
            channel_id="test_channel",
        )
        print(f"✅ Created timer via TimerManager: {timer_id}")

        # List timers
        timers = await timer_manager.list_timers(user_id="test_user")
        print(f"✅ Found {len(timers)} timers via TimerManager")

        # Cancel timer
        cancelled = await timer_manager.cancel_timer(timer_id)
        print(f"✅ Cancelled timer: {cancelled}")

    except Exception as e:
        print(f"❌ Error testing timer lifecycle: {e}")
        import traceback

        traceback.print_exc()


def test_shell_tool():
    """Test shell tool with non-interactive mode."""
    print("\n=== Testing Shell Tool Non-Interactive Mode ===")

    try:
        from strands_tools import shell

        # Test simple command
        print("\n1. Testing simple shell command...")
        result = shell(command="echo 'Hello World'", non_interactive=True)
        print(f"Shell result: {result}")

        if result.get("status") == "success":
            print("✅ Shell tool works in non-interactive mode")
        else:
            print(f"❌ Shell tool failed: {result}")

    except Exception as e:
        print(f"❌ Error testing shell tool: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests."""
    print("🧪 Timer System Diagnostic Test")
    print("=" * 50)

    await test_timer_tools()
    await test_timer_lifecycle()
    test_shell_tool()

    print("\n" + "=" * 50)
    print("✅ Diagnostic test completed!")


if __name__ == "__main__":
    asyncio.run(main())
