"""Test Heartbeat System

Verifies that:
1. Message bus starts correctly
2. Heartbeat tasks are created
3. FAST_HEARTBEAT_TICK events are published
4. Timer role receives heartbeat events
"""

import asyncio
import logging

# Enable debug logging to see heartbeat ticks
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_heartbeat():
    print("=" * 70)
    print("HEARTBEAT SYSTEM TEST")
    print("=" * 70)

    from supervisor.supervisor import Supervisor

    print("\n[1/5] Initializing Supervisor...")
    supervisor = Supervisor(None)  # Use default config

    print("\n[2/5] Starting Supervisor (starts message bus)...")
    supervisor.start()

    # Check if message bus is running
    if supervisor.message_bus and supervisor.message_bus.is_running():
        print("  ✓ Message bus is running")
    else:
        print("  ✗ Message bus is NOT running!")
        return 1

    print("\n[3/5] Starting async tasks (creates heartbeat)...")
    await supervisor.start_async_tasks()

    print("\n[4/5] Subscribing to FAST_HEARTBEAT_TICK events...")

    # Create a counter to track heartbeat ticks
    tick_counter = {"count": 0}

    def heartbeat_listener(message):
        tick_counter["count"] += 1
        tick = message.get("tick", "?")
        timestamp = message.get("timestamp", 0)
        print(f"  ✓ Received FAST_HEARTBEAT_TICK #{tick} at {timestamp:.2f}")

    # Subscribe to heartbeat events
    supervisor.message_bus.subscribe("FAST_HEARTBEAT_TICK", heartbeat_listener)

    print("\n[5/5] Waiting for 3 heartbeat ticks (15 seconds)...")
    print("  (Heartbeat interval is 5 seconds)")

    # Wait for 16 seconds to see 3 ticks
    for i in range(16):
        await asyncio.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}s elapsed, expecting tick #{(i+1)//5}")

    print(f"\n✓ Received {tick_counter['count']} heartbeat ticks in 16 seconds")

    if tick_counter["count"] >= 3:
        print("\n" + "=" * 70)
        print("✓ HEARTBEAT SYSTEM IS WORKING")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ HEARTBEAT SYSTEM NOT WORKING")
        print("=" * 70)
        print(f"Expected at least 3 ticks, got {tick_counter['count']}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(test_heartbeat())
        exit(exit_code)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
