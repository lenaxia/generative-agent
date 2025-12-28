"""
Test Phase 4 SimplifiedWorkflowEngine.

Tests the simplified workflow engine to ensure it can be instantiated
and has the correct structure before full integration.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

from common.agent_configuration import AgentConfiguration
from common.intent_collector import IntentCollector
from common.intent_processor import IntentProcessor
from common.intents import NotificationIntent
from llm_provider.runtime_agent_factory import RuntimeAgentFactory
from supervisor.simplified_workflow_engine import (
    SimplifiedWorkflowEngine,
    WorkflowResult,
)


def test_workflow_result_creation():
    """Test WorkflowResult dataclass creation."""
    print("\n[TEST 1] WorkflowResult Creation")
    print("=" * 60)

    # Success result
    result = WorkflowResult(
        response="Task completed successfully",
        metadata={"tool_count": 3, "intent_count": 2},
        success=True,
    )

    assert result.response == "Task completed successfully"
    assert result.success is True
    assert result.error is None
    print("✓ Success result created")

    # Error result
    error_result = WorkflowResult(
        response="Error occurred",
        success=False,
        error="Something went wrong",
    )

    assert error_result.success is False
    assert error_result.error == "Something went wrong"
    print("✓ Error result created")

    return True


def test_engine_initialization():
    """Test SimplifiedWorkflowEngine initialization."""
    print("\n[TEST 2] SimplifiedWorkflowEngine Initialization")
    print("=" * 60)

    # Mock dependencies
    agent_factory = Mock(spec=RuntimeAgentFactory)
    intent_processor = Mock(spec=IntentProcessor)

    # Create engine
    engine = SimplifiedWorkflowEngine(
        agent_factory=agent_factory,
        intent_processor=intent_processor,
    )

    assert engine.agent_factory == agent_factory
    assert engine.intent_processor == intent_processor
    print("✓ Engine initialized with dependencies")

    # Check status
    status = engine.get_status()
    assert status["engine_type"] == "SimplifiedWorkflowEngine"
    assert status["phase"] == 4
    assert status["features"]["meta_planning"] is True
    assert status["features"]["dag_execution"] is False
    print(f"✓ Engine status: {status['engine_type']}, Phase {status['phase']}")

    return True


async def test_execute_complex_request_success():
    """Test execute_complex_request with successful execution."""
    print("\n[TEST 3] Execute Complex Request (Success)")
    print("=" * 60)

    # Mock agent factory
    agent_factory = Mock(spec=RuntimeAgentFactory)
    mock_intent_collector = IntentCollector()

    # Mock agent result
    mock_agent_result = Mock()
    mock_agent_result.final_output = (
        "Weather is sunny, calendar is clear. Perfect for hiking!"
    )

    # Mock agent
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=mock_agent_result)

    # Configure factory to return mocked agent
    agent_factory.create_agent = Mock(return_value=(mock_agent, mock_intent_collector))

    # Mock intent processor
    intent_processor = Mock(spec=IntentProcessor)
    intent_processor.process_intents = AsyncMock()

    # Create engine
    engine = SimplifiedWorkflowEngine(
        agent_factory=agent_factory,
        intent_processor=intent_processor,
    )

    # Create test configuration
    config = AgentConfiguration(
        plan="1. Check weather\\n2. Check calendar\\n3. Suggest activities",
        system_prompt="You combine weather and calendar data.",
        tool_names=["weather.get_forecast", "calendar.list_events"],
        guidance="Be concise",
        max_iterations=10,
        metadata={},
    )

    # Execute workflow
    result = await engine.execute_complex_request(
        request="What should I do this weekend?",
        agent_config=config,
        context=None,
    )

    # Verify result
    assert result.success is True
    assert result.error is None
    assert "sunny" in result.response
    assert result.metadata["tool_count"] == 2
    print(f"✓ Workflow executed successfully")
    print(f"  Response: {result.response[:60]}...")
    print(f"  Tools used: {result.metadata['tools_used']}")

    # Verify mocks were called
    agent_factory.create_agent.assert_called_once()
    mock_agent.run.assert_called_once()
    print(f"✓ All components called correctly")

    return True


async def test_execute_complex_request_with_intents():
    """Test execute_complex_request with intents collected."""
    print("\n[TEST 4] Execute Complex Request (With Intents)")
    print("=" * 60)

    # Mock agent factory
    agent_factory = Mock(spec=RuntimeAgentFactory)
    mock_intent_collector = IntentCollector()

    # Add some test intents
    mock_intent_collector.register(
        NotificationIntent(
            message="Weather alert",
            channel="test_channel",
        )
    )

    # Mock agent result
    mock_agent_result = Mock()
    mock_agent_result.final_output = "Task completed with notification"

    # Mock agent
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=mock_agent_result)

    # Configure factory
    agent_factory.create_agent = Mock(return_value=(mock_agent, mock_intent_collector))

    # Mock intent processor
    intent_processor = Mock(spec=IntentProcessor)
    intent_processor.process_intents = AsyncMock()

    # Create engine
    engine = SimplifiedWorkflowEngine(
        agent_factory=agent_factory,
        intent_processor=intent_processor,
    )

    # Create test configuration
    config = AgentConfiguration(
        plan="Check weather and notify",
        system_prompt="You check weather and send alerts.",
        tool_names=["weather.get_forecast"],
        guidance="Send notifications",
        max_iterations=5,
        metadata={},
    )

    # Execute workflow
    result = await engine.execute_complex_request(
        request="Check weather and notify me",
        agent_config=config,
        context=None,
    )

    # Verify result
    assert result.success is True
    assert result.metadata["intent_count"] == 1
    print(f"✓ Workflow executed with {result.metadata['intent_count']} intent(s)")

    # Verify intent processor was called
    intent_processor.process_intents.assert_called_once()
    print(f"✓ Intents processed")

    return True


async def test_execute_complex_request_error():
    """Test execute_complex_request with error."""
    print("\n[TEST 5] Execute Complex Request (Error Handling)")
    print("=" * 60)

    # Mock agent factory that raises error
    agent_factory = Mock(spec=RuntimeAgentFactory)
    agent_factory.create_agent = Mock(side_effect=ValueError("Invalid configuration"))

    # Mock intent processor
    intent_processor = Mock(spec=IntentProcessor)

    # Create engine
    engine = SimplifiedWorkflowEngine(
        agent_factory=agent_factory,
        intent_processor=intent_processor,
    )

    # Create test configuration
    config = AgentConfiguration(
        plan="Test error handling",
        system_prompt="This will fail",
        tool_names=["nonexistent.tool"],
        guidance="",
        max_iterations=5,
        metadata={},
    )

    # Execute workflow (should handle error gracefully)
    result = await engine.execute_complex_request(
        request="This should fail",
        agent_config=config,
        context=None,
    )

    # Verify error result
    assert result.success is False
    assert result.error is not None
    assert "Invalid configuration" in result.error
    print(f"✓ Error handled gracefully")
    print(f"  Error: {result.error}")

    return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE 4 SIMPLIFIED WORKFLOW ENGINE TESTS")
    print("=" * 60)

    tests = [
        ("WorkflowResult Creation", test_workflow_result_creation),
        ("Engine Initialization", test_engine_initialization),
        ("Execute Complex Request (Success)", test_execute_complex_request_success),
        (
            "Execute Complex Request (With Intents)",
            test_execute_complex_request_with_intents,
        ),
        ("Execute Complex Request (Error)", test_execute_complex_request_error),
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
        print("\n✅ ALL TESTS PASSED - SimplifiedWorkflowEngine is working correctly!")
        return True
    else:
        print(f"\n❌ {total - passed} TESTS FAILED - Please review errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
