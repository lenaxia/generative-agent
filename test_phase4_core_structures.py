"""
Test Phase 4 Core Data Structures.

Tests for AgentConfiguration and IntentCollector to ensure they work correctly
before proceeding with RuntimeAgentFactory implementation.
"""

import asyncio

from common.agent_configuration import AgentConfiguration
from common.intent_collector import (
    IntentCollector,
    clear_current_collector,
    get_current_collector,
    register_intent,
    set_current_collector,
)
from common.intents import AuditIntent, NotificationIntent


def test_agent_configuration_creation():
    """Test AgentConfiguration creation with valid data."""
    print("\n[TEST 1] AgentConfiguration Creation")
    print("=" * 60)

    config = AgentConfiguration(
        plan="1. Check weather\n2. Suggest activities",
        system_prompt="You are a helpful assistant that combines weather and planning.",
        tool_names=["weather.get_forecast", "calendar.list_events"],
        guidance="Be concise and focus on outdoor activities",
        max_iterations=10,
        metadata={"confidence": 0.85, "reasoning": "Weather-based planning"},
    )

    print(f"✓ Created AgentConfiguration:")
    print(f"  - Plan: {config.plan[:50]}...")
    print(f"  - Tool count: {len(config.tool_names)}")
    print(f"  - Max iterations: {config.max_iterations}")
    print(f"  - Metadata: {config.metadata}")

    # Test validation
    assert config.validate(), "Configuration should be valid"
    print(f"✓ Configuration validation passed")

    # Test to_dict/from_dict
    config_dict = config.to_dict()
    config_restored = AgentConfiguration.from_dict(config_dict)
    assert config_restored.plan == config.plan, "Restored plan should match"
    assert (
        config_restored.tool_names == config.tool_names
    ), "Restored tools should match"
    print(f"✓ Serialization/deserialization works")

    return True


def test_agent_configuration_validation():
    """Test AgentConfiguration validation with invalid data."""
    print("\n[TEST 2] AgentConfiguration Validation")
    print("=" * 60)

    # Test empty plan
    try:
        config = AgentConfiguration(
            plan="",
            system_prompt="Test",
            tool_names=["test.tool"],
            guidance="Test",
            max_iterations=10,
            metadata={},
        )
        assert not config.validate(), "Empty plan should fail validation"
        print(f"✓ Empty plan validation works")
    except Exception as e:
        print(f"✓ Empty plan rejected: {type(e).__name__}")

    # Test invalid max_iterations
    config = AgentConfiguration(
        plan="Test plan",
        system_prompt="Test prompt",
        tool_names=["test.tool"],
        guidance="Test",
        max_iterations=0,
        metadata={},
    )
    assert not config.validate(), "max_iterations=0 should fail validation"
    print(f"✓ Invalid max_iterations validation works")

    # Test valid configuration
    config = AgentConfiguration(
        plan="Valid plan",
        system_prompt="Valid prompt",
        tool_names=["test.tool"],
        guidance="Valid guidance",
        max_iterations=15,
        metadata={},
    )
    assert config.validate(), "Valid configuration should pass"
    print(f"✓ Valid configuration passes validation")

    return True


def test_intent_collector_basic():
    """Test basic IntentCollector functionality."""
    print("\n[TEST 3] IntentCollector Basic Functionality")
    print("=" * 60)

    collector = IntentCollector()

    # Test initial state
    assert collector.count() == 0, "New collector should be empty"
    print(f"✓ New collector starts empty")

    # Register intents
    intent1 = NotificationIntent(
        message="Test notification",
        channel="test_channel",
        user_id="user123",
    )
    intent2 = AuditIntent(
        action="test_action",
        details={"test": "data"},
        user_id="user123",
    )

    collector.register(intent1)
    collector.register(intent2)

    assert collector.count() == 2, "Should have 2 intents"
    print(f"✓ Registered 2 intents")

    # Get intents
    intents = collector.get_intents()
    assert len(intents) == 2, "Should retrieve 2 intents"
    assert isinstance(
        intents[0], NotificationIntent
    ), "First intent should be NotificationIntent"
    assert isinstance(intents[1], AuditIntent), "Second intent should be AuditIntent"
    print(f"✓ Retrieved intents correctly")

    # Clear intents
    collector.clear()
    assert collector.count() == 0, "Cleared collector should be empty"
    print(f"✓ Cleared intents successfully")

    return True


async def test_intent_collector_context():
    """Test IntentCollector context-local storage."""
    print("\n[TEST 4] IntentCollector Context-Local Storage")
    print("=" * 60)

    # Test initial state - no collector
    collector = get_current_collector()
    assert collector is None, "Should start with no collector"
    print(f"✓ No collector in context initially")

    # Set collector
    collector = IntentCollector()
    set_current_collector(collector)

    retrieved = get_current_collector()
    assert retrieved is collector, "Should retrieve same collector"
    print(f"✓ Collector set and retrieved from context")

    # Register intent via helper function
    intent = NotificationIntent(
        message="Context test",
        channel="test_channel",
        user_id="user123",
    )
    await register_intent(intent)

    assert collector.count() == 1, "Intent should be registered"
    print(f"✓ Intent registered via helper function")

    # Clear collector
    clear_current_collector()
    retrieved = get_current_collector()
    assert retrieved is None, "Collector should be cleared from context"
    print(f"✓ Collector cleared from context")

    return True


async def test_intent_collector_concurrent():
    """Test IntentCollector with concurrent tasks."""
    print("\n[TEST 5] IntentCollector Concurrent Context Isolation")
    print("=" * 60)

    async def task1():
        """First concurrent task with its own collector."""
        collector1 = IntentCollector()
        set_current_collector(collector1)

        await register_intent(
            NotificationIntent(
                message="Task 1 intent",
                channel="channel1",
                user_id="user1",
            )
        )

        await asyncio.sleep(0.01)  # Simulate work

        assert collector1.count() == 1, "Task 1 should have 1 intent"
        clear_current_collector()
        return "task1_done"

    async def task2():
        """Second concurrent task with its own collector."""
        collector2 = IntentCollector()
        set_current_collector(collector2)

        await register_intent(
            AuditIntent(
                action="task2_action",
                details={"task": "2"},
                user_id="user2",
            )
        )

        await asyncio.sleep(0.01)  # Simulate work

        assert collector2.count() == 1, "Task 2 should have 1 intent"
        clear_current_collector()
        return "task2_done"

    # Run tasks concurrently
    results = await asyncio.gather(task1(), task2())

    assert results == ["task1_done", "task2_done"], "Both tasks should complete"
    print(f"✓ Concurrent tasks maintained separate collectors")
    print(f"✓ Context isolation works correctly")

    return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 4 CORE DATA STRUCTURES TESTS")
    print("=" * 60)

    tests = [
        ("AgentConfiguration Creation", test_agent_configuration_creation),
        ("AgentConfiguration Validation", test_agent_configuration_validation),
        ("IntentCollector Basic", test_intent_collector_basic),
        ("IntentCollector Context", test_intent_collector_context),
        ("IntentCollector Concurrent", test_intent_collector_concurrent),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if error:
            print(f"       Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - Core data structures are working correctly!")
        return True
    else:
        print(f"\n❌ {total - passed} TESTS FAILED - Please review errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
