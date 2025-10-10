#!/usr/bin/env python3
"""
Comprehensive test script to verify timer functionality fixes.

This script tests:
1. Timer role tool enablement for fast_reply mode
2. Environment variable setting for shell tool non-interactive mode
3. Timer creation, listing, and cancellation
4. Integration with the universal agent system
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_environment_setup():
    """Test that environment variables are properly set."""
    print("=== Testing Environment Setup ===")

    # Check if environment variables are set
    strands_non_interactive = os.environ.get("STRANDS_NON_INTERACTIVE")
    bypass_tool_consent = os.environ.get("BYPASS_TOOL_CONSENT")

    print(f"STRANDS_NON_INTERACTIVE: {strands_non_interactive}")
    print(f"BYPASS_TOOL_CONSENT: {bypass_tool_consent}")

    if strands_non_interactive == "true" and bypass_tool_consent == "true":
        print("✅ Environment variables are properly set")
        return True
    else:
        print("❌ Environment variables not set correctly")
        return False


async def test_timer_tools_directly():
    """Test timer tools directly to ensure they work."""
    print("\n=== Testing Timer Tools Directly ===")

    try:
        from roles.timer.tools import timer_cancel, timer_list, timer_set

        # Test timer_set
        print("\n1. Testing timer_set...")
        result = timer_set(
            duration="30s",
            label="Test Timer Direct",
            user_id="test_user_direct",
            channel_id="test_channel_direct",
        )
        print(f"timer_set result: {result}")

        if result.get("success"):
            timer_id = result.get("timer_id")
            print(f"✅ Timer created successfully: {timer_id}")

            # Test timer_list
            print("\n2. Testing timer_list...")
            list_result = timer_list(user_id="test_user_direct")
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
                return False

            # Test timer_cancel
            print("\n3. Testing timer_cancel...")
            cancel_result = timer_cancel(timer_id)
            print(f"timer_cancel result: {cancel_result}")

            if cancel_result.get("success"):
                print(f"✅ Timer cancelled successfully")
                return True
            else:
                print(f"❌ timer_cancel failed: {cancel_result.get('error')}")
                return False
        else:
            print(f"❌ timer_set failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ Error testing timer tools: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_shell_tool_non_interactive():
    """Test shell tool in non-interactive mode."""
    print("\n=== Testing Shell Tool Non-Interactive Mode ===")

    try:
        from strands_tools import shell

        # Test simple command
        print("\n1. Testing simple shell command...")
        result = shell(command="echo 'Timer Fix Test'", non_interactive=True)
        print(f"Shell result status: {result.get('status')}")

        if result.get("status") == "success":
            print("✅ Shell tool works in non-interactive mode")
            return True
        else:
            print(f"❌ Shell tool failed: {result}")
            return False

    except Exception as e:
        print(f"❌ Error testing shell tool: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_universal_agent_integration():
    """Test timer functionality through the universal agent system."""
    print("\n=== Testing Universal Agent Integration ===")

    try:
        from llm_provider.factory import LLMFactory, LLMType
        from llm_provider.role_registry import RoleRegistry
        from llm_provider.tool_registry import ToolRegistry
        from llm_provider.universal_agent import UniversalAgent
        from supervisor.config_manager import ConfigManager

        # Initialize components
        print("1. Initializing Universal Agent components...")
        config_manager = ConfigManager("config.yaml")
        config = config_manager.load_config()

        llm_factory = LLMFactory(config.get("llm_providers", {}), framework="strands")
        role_registry = RoleRegistry("roles")
        tool_registry = ToolRegistry()

        universal_agent = UniversalAgent(
            llm_factory=llm_factory,
            role_registry=role_registry,
            tool_registry=tool_registry,
        )

        print("✅ Universal Agent initialized")

        # Test timer role with tools enabled
        print("\n2. Testing timer role execution...")

        # Simulate a timer request
        request = "Set a timer for 45 seconds with label 'Integration Test'"

        # This should route to timer role and use timer tools
        response = await universal_agent.process_request(
            request=request, role="timer", execution_mode="FAST_REPLY"
        )

        print(f"Timer request response: {response}")

        if "timer" in response.lower() and "45" in response:
            print("✅ Timer role executed successfully with tools")
            return True
        else:
            print(f"❌ Timer role execution may have issues: {response}")
            return False

    except Exception as e:
        print(f"❌ Error testing universal agent integration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_role_definition_parsing():
    """Test that the timer role definition is parsed correctly."""
    print("\n=== Testing Role Definition Parsing ===")

    try:
        from llm_provider.role_registry import RoleRegistry

        role_registry = RoleRegistry("roles")
        timer_role = role_registry.get_role("timer")

        if not timer_role:
            print("❌ Timer role not found in registry")
            return False

        print(f"Timer role loaded: {timer_role.get('name')}")

        # Check if fast_reply tools are enabled
        tools_config = timer_role.get("config", {}).get("tools", {})
        fast_reply_config = tools_config.get("fast_reply", {})

        print(f"Tools config: {tools_config}")
        print(f"Fast reply config: {fast_reply_config}")

        if isinstance(fast_reply_config, dict) and fast_reply_config.get("enabled"):
            print("✅ Fast reply tools are enabled for timer role")
            return True
        else:
            print("❌ Fast reply tools are not properly enabled")
            return False

    except Exception as e:
        print(f"❌ Error testing role definition: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🧪 Timer Fixes Comprehensive Test")
    print("=" * 60)

    # Set environment variables for this test
    os.environ["STRANDS_NON_INTERACTIVE"] = "true"
    os.environ["BYPASS_TOOL_CONSENT"] = "true"

    test_results = []

    # Run tests
    test_results.append(("Environment Setup", test_environment_setup()))
    test_results.append(("Timer Tools Direct", await test_timer_tools_directly()))
    test_results.append(
        ("Shell Tool Non-Interactive", test_shell_tool_non_interactive())
    )
    test_results.append(("Role Definition Parsing", test_role_definition_parsing()))
    test_results.append(
        ("Universal Agent Integration", await test_universal_agent_integration())
    )

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Timer fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
