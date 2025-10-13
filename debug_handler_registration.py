#!/usr/bin/env python3
"""
Debug script to demonstrate the Slack handler registration issue.
This script shows why timer notifications fail to reach Slack channels.
"""

import logging
import os
import sys

# Add the project root to Python path
sys.path.insert(0, "/home/mikekao/personal/generative-agent")

# Configure logging to see debug output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def demonstrate_handler_registration_issue():
    """Demonstrate the Slack handler registration and routing issue."""

    print("=== SLACK HANDLER REGISTRATION ANALYSIS ===")

    from common.channel_handlers.console_handler import ConsoleChannelHandler
    from common.channel_handlers.slack_handler import SlackChannelHandler
    from common.communication_manager import CommunicationManager
    from common.message_bus import MessageBus

    # Create communication manager (like in timer expiry)
    message_bus = MessageBus()
    message_bus.start()
    comm_manager = CommunicationManager(message_bus)

    print("\n1. INITIAL STATE - No Handlers Registered")
    print(f"   Registered channels: {list(comm_manager.channels.keys())}")

    # Test channel lookup with no handlers
    handler = comm_manager._find_handler_for_channel("slack:C52L1UK5E")
    print(f"   Handler for 'slack:C52L1UK5E': {handler}")

    print("\n2. MANUAL HANDLER REGISTRATION (What's Missing)")
    # This is what should happen during system initialization
    slack_handler = SlackChannelHandler()
    console_handler = ConsoleChannelHandler()

    # Register handlers manually (this is missing in timer expiry)
    comm_manager.register_channel(slack_handler)
    comm_manager.register_channel(console_handler)

    print(f"   Registered channels: {list(comm_manager.channels.keys())}")
    print(
        f"   Slack handler pattern: {getattr(slack_handler, 'channel_pattern', 'NOT SET')}"
    )

    print("\n3. PATTERN MATCHING TEST")
    # Test pattern matching (the fix)
    test_channels = ["slack:C52L1UK5E", "slack:#general", "console", "unknown:channel"]

    for channel in test_channels:
        handler = comm_manager._find_handler_for_channel(channel)
        handler_name = handler.__class__.__name__ if handler else "None"
        print(f"   '{channel}' → {handler_name}")

    print("\n4. TARGET CHANNEL DETERMINATION")
    # Test the routing logic
    origin_channel = "slack:C52L1UK5E"
    message_type = "notification"
    context = {}

    targets = comm_manager._determine_target_channels(
        origin_channel, message_type, context
    )
    print(f"   Origin: '{origin_channel}' → Targets: {targets}")

    print("\n5. COMPLETE ROUTING TEST")
    # Test the complete routing flow
    for target in targets:
        handler = comm_manager._find_handler_for_channel(target)
        if handler:
            print(f"   ✅ '{target}' → {handler.__class__.__name__}")
        else:
            print(f"   ❌ '{target}' → No handler found")

    print("\n=== ROOT CAUSE ANALYSIS ===")
    print(
        "❌ PROBLEM: Timer expiry creates CommunicationManager without handler registration"
    )
    print("❌ ISSUE: CommunicationManager.get_instance() creates empty manager")
    print("❌ RESULT: No handlers available, notifications lost")
    print()
    print("✅ SOLUTION: Ensure handlers are registered during system initialization")
    print(
        "✅ FIX: Use supervisor's CommunicationManager instance with registered handlers"
    )
    print("✅ PATTERN: Slack handler uses 'slack:' pattern to match 'slack:C52L1UK5E'")


if __name__ == "__main__":
    demonstrate_handler_registration_issue()
