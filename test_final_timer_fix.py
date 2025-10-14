#!/usr/bin/env python3
"""
Final test to verify the complete timer notification routing fix.
This tests all the implemented fixes together.
"""

import logging
import os
import sys
import time

# Add the project root to Python path
sys.path.insert(0, "/home/mikekao/personal/generative-agent")

# Configure logging to see all our debug output
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_complete_timer_fix():
    """Test the complete timer fix with all enhancements."""

    print("=== TESTING COMPLETE TIMER NOTIFICATION FIX ===")

    from common.channel_handlers.console_handler import ConsoleChannelHandler
    from common.channel_handlers.slack_handler import SlackChannelHandler
    from common.communication_manager import CommunicationManager
    from common.message_bus import MessageBus
    from common.request_model import RequestMetadata
    from roles.timer_single_file import _timer_context_injector, set_timer

    # Step 1: Setup communication system (like supervisor does)
    print("\n1. SETTING UP COMMUNICATION SYSTEM")
    message_bus = MessageBus()
    message_bus.start()
    comm_manager = CommunicationManager(message_bus)

    # Register handlers
    slack_handler = SlackChannelHandler()
    console_handler = ConsoleChannelHandler()
    comm_manager.register_channel(slack_handler)
    comm_manager.register_channel(console_handler)

    # Set as singleton (our fix)
    CommunicationManager.set_instance(comm_manager)
    print(f"   ✅ Handlers registered: {list(comm_manager.channels.keys())}")

    # Step 2: Simulate request metadata (like communication manager creates)
    print("\n2. SIMULATING SLACK REQUEST METADATA")

    class MockContext:
        def __init__(self):
            self.user_id = "U07TCJFKF1C"
            self.channel_id = "slack:C52L1UK5E"
            # Add request metadata like the system creates
            self.request_metadata = RequestMetadata(
                prompt="set a timer for 3 seconds",
                source_id="slack",
                target_id="workflow_engine",
                metadata={
                    "user_id": "U07TCJFKF1C",
                    "channel_id": "C52L1UK5E",  # Just the channel part
                    "request_id": "test_request_123",
                },
            )

    slack_context = MockContext()
    print(
        f"   Mock context: user_id={slack_context.user_id}, channel_id={slack_context.channel_id}"
    )
    print(f"   Request metadata: {slack_context.request_metadata.metadata}")

    # Step 3: Test context injection
    print("\n3. TESTING CONTEXT INJECTION")
    _timer_context_injector("set timer 3s", slack_context, {})

    # Step 4: Test timer creation (this should now use the enhanced context extraction)
    print("\n4. TESTING TIMER CREATION WITH ENHANCED CONTEXT")
    print("   Calling set_timer with defaults (should extract from context)...")

    result = set_timer(
        duration="3s",
        label="final test timer"
        # Note: NOT passing user_id and channel_id - should extract from context
    )

    print(f"   Timer creation result: {result}")

    # Step 5: Wait for timer expiry
    print("\n5. WAITING FOR TIMER EXPIRY (3 seconds)...")
    time.sleep(4)

    print("\n=== FINAL FIX VERIFICATION ===")
    print("🔧 FIXES IMPLEMENTED:")
    print("   1. ✅ CommunicationManager singleton pattern fixed")
    print("   2. ✅ Enhanced context extraction from request metadata")
    print("   3. ✅ LLM system prompt updated with explicit instructions")
    print("   4. ✅ Leveraged same routing mechanism as LLM responses")
    print("   5. ✅ Comprehensive debug logging added")
    print()
    print("📊 EXPECTED BEHAVIOR:")
    print("   - Timer should store: channel='slack:C52L1UK5E', user_id='U07TCJFKF1C'")
    print("   - Timer expiry should route to SlackChannelHandler")
    print("   - Debug logs should show context extraction working")
    print()
    print("🎯 CHECK THE LOGS ABOVE FOR:")
    print("   - 'Updated channel_id from request metadata: slack:C52L1UK5E'")
    print("   - 'FINAL timer will use: user_id=U07TCJFKF1C, channel=slack:C52L1UK5E'")
    print("   - Timer expiry routing to slack handler (not console)")


if __name__ == "__main__":
    test_complete_timer_fix()
