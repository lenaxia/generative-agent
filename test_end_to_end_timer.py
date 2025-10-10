#!/usr/bin/env python3
"""
End-to-end test for timer notification flow.
Tests the complete flow from timer creation to Slack notification.
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

# Set environment variables
os.environ["STRANDS_NON_INTERACTIVE"] = "true"
os.environ["BYPASS_TOOL_CONSENT"] = "true"


async def test_timer_with_context():
    """Test timer creation with proper Slack context."""
    print("=== Testing Timer with Slack Context ===")

    try:
        from roles.timer.lifecycle import parse_timer_parameters

        # Simulate Slack context
        context = {
            "user_id": "U52L1U8M6",  # Slack user ID from logs
            "channel_id": "C1234567890",  # Slack channel ID
        }

        # Simulate router parameters
        parameters = {
            "action": "set",
            "duration": "10s",  # Short timer for testing
        }

        print(f"Testing with context: {context}")
        print(f"Testing with parameters: {parameters}")

        # Call parse_timer_parameters (this is what happens in pre-processing)
        result = await parse_timer_parameters(
            instruction="set a 10 second timer", context=context, parameters=parameters
        )

        print(f"Parse result: {result}")

        execution_result = result.get("execution_result", {})
        if execution_result.get("success"):
            timer_id = execution_result.get("timer_id")
            print(f"✅ Timer created with context: {timer_id}")

            # Check that the timer has the correct context
            from roles.timer.lifecycle import get_timer_manager

            timer_manager = get_timer_manager()
            timer_data = await timer_manager.get_timer(timer_id)

            if timer_data:
                print(f"Timer user_id: {timer_data.get('user_id')}")
                print(f"Timer channel_id: {timer_data.get('channel_id')}")

                if timer_data.get("channel_id").startswith("slack:"):
                    print("✅ Timer has correct Slack channel format")
                    return timer_id
                else:
                    print("❌ Timer channel_id format incorrect")
                    return None
            else:
                print("❌ Could not retrieve timer data")
                return None
        else:
            print(f"❌ Timer creation failed: {execution_result}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_message_bus_subscription():
    """Test that timer events can be published and received."""
    print("\n=== Testing MessageBus Timer Events ===")

    try:
        from common.message_bus import MessageBus, MessageType

        # Create message bus
        message_bus = MessageBus()
        message_bus.start()

        # Create a test subscriber
        received_events = []

        def test_handler(event_data):
            print(f"📨 Received timer event: {event_data}")
            received_events.append(event_data)

        # Subscribe to timer events
        message_bus.subscribe(None, MessageType.TIMER_EXPIRED, test_handler)

        # Publish a test timer event
        test_event = {
            "timer_id": "test_timer_123",
            "timer_name": "Test Timer",
            "custom_message": "Test timer expired!",
            "user_id": "U52L1U8M6",
            "channel_id": "slack:C1234567890",
        }

        message_bus.publish(None, MessageType.TIMER_EXPIRED, test_event)

        # Give it a moment to process
        time.sleep(0.1)

        if received_events:
            print(f"✅ MessageBus working: received {len(received_events)} events")
            return True
        else:
            print("❌ MessageBus not working: no events received")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_timer_monitor():
    """Test that timer monitor can find and process expired timers."""
    print("\n=== Testing Timer Monitor ===")

    try:
        from common.message_bus import MessageBus
        from supervisor.timer_monitor import TimerMonitor

        # Create message bus and timer monitor
        message_bus = MessageBus()
        message_bus.start()

        timer_monitor = TimerMonitor(message_bus)

        # Create a timer that expires immediately
        from roles.timer.lifecycle import get_timer_manager

        timer_manager = get_timer_manager()

        timer_id = await timer_manager.create_timer(
            timer_type="countdown",
            duration_seconds=1,  # 1 second timer
            name="Monitor Test Timer",
            user_id="U52L1U8M6",
            channel_id="slack:C1234567890",
        )

        print(f"Created test timer: {timer_id}")

        # Wait for timer to expire
        await asyncio.sleep(2)

        # Check for expired timers
        expired_timers = await timer_monitor.check_expired_timers()
        print(f"Found {len(expired_timers)} expired timers")

        if expired_timers:
            # Process the expired timer
            for timer in expired_timers:
                await timer_monitor.process_expired_timer(timer)
            print("✅ Timer monitor processed expired timers")
            return True
        else:
            print("❌ No expired timers found")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run end-to-end tests."""
    print("🧪 End-to-End Timer Notification Test")
    print("=" * 60)

    test_results = []

    # Test timer creation with context
    timer_id = await test_timer_with_context()
    test_results.append(("Timer Creation with Context", timer_id is not None))

    # Test message bus
    test_results.append(("MessageBus Events", test_message_bus_subscription()))

    # Test timer monitor
    test_results.append(("Timer Monitor", await test_timer_monitor()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All end-to-end tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
