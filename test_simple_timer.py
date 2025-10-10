#!/usr/bin/env python3
"""
Simple test to verify timer functionality fixes.
"""

import os
import sys

# Set environment variables to disable prompts
os.environ["STRANDS_NON_INTERACTIVE"] = "true"
os.environ["BYPASS_TOOL_CONSENT"] = "true"


def test_timer_tools():
    """Test timer tools directly."""
    print("=== Testing Timer Tools ===")

    try:
        from roles.timer.tools import timer_cancel, timer_list, timer_set

        # Test timer_set
        print("1. Testing timer_set...")
        result = timer_set(
            duration="30s",
            label="Simple Test Timer",
            user_id="test_user",
            channel_id="test_channel",
        )
        print(f"Result: {result}")

        if result.get("success"):
            timer_id = result.get("timer_id")
            print(f"✅ Timer created: {timer_id}")

            # Test timer_list
            print("\n2. Testing timer_list...")
            list_result = timer_list(user_id="test_user")
            print(f"List result: {list_result}")

            if list_result.get("success"):
                print(f"✅ Found {len(list_result.get('timers', []))} timers")

            # Test timer_cancel
            print("\n3. Testing timer_cancel...")
            cancel_result = timer_cancel(timer_id)
            print(f"Cancel result: {cancel_result}")

            if cancel_result.get("success"):
                print("✅ Timer cancelled successfully")
                return True

        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_role_config():
    """Test that timer role has tools enabled."""
    print("\n=== Testing Timer Role Configuration ===")

    try:
        import yaml

        with open("roles/timer/definition.yaml") as f:
            timer_def = yaml.safe_load(f)

        tools_config = timer_def.get("tools", {})
        fast_reply = tools_config.get("fast_reply", {})

        print(f"Tools config: {tools_config}")

        if isinstance(fast_reply, dict) and not fast_reply.get("enabled"):
            print("✅ Fast reply tools are disabled (using pre-processing instead)")
            return True
        else:
            print("❌ Fast reply tools should be disabled for pre-processing approach")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_environment():
    """Test environment variables."""
    print("\n=== Testing Environment Variables ===")

    strands_non_interactive = os.environ.get("STRANDS_NON_INTERACTIVE")
    bypass_tool_consent = os.environ.get("BYPASS_TOOL_CONSENT")

    print(f"STRANDS_NON_INTERACTIVE: {strands_non_interactive}")
    print(f"BYPASS_TOOL_CONSENT: {bypass_tool_consent}")

    if strands_non_interactive == "true" and bypass_tool_consent == "true":
        print("✅ Environment variables set correctly")
        return True
    else:
        print("❌ Environment variables not set")
        return False


def main():
    """Run tests."""
    print("🧪 Simple Timer Test")
    print("=" * 40)

    results = []
    results.append(("Environment", test_environment()))
    results.append(("Role Config", test_role_config()))
    results.append(("Timer Tools", test_timer_tools()))

    print("\n" + "=" * 40)
    print("Results:")

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<15} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
