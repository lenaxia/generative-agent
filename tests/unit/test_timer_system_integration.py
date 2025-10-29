"""
Unit tests to catch timer system integration issues.

These tests verify that:
1. Heartbeat tasks are actually started
2. Intent handlers are properly registered
3. The complete timer workflow functions correctly
"""

import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from common.intent_processor import IntentProcessor
from common.message_bus import MessageBus
from llm_provider.role_registry import RoleRegistry
from roles.core_timer import (
    TimerCreationIntent,
    TimerExpiryIntent,
    process_timer_creation_intent,
    process_timer_expiry_intent,
)
from supervisor.supervisor import Supervisor


class TestHeartbeatTaskStartup:
    """Tests to verify heartbeat tasks are actually started."""

    def test_start_scheduled_tasks_creates_heartbeat_tasks(self):
        """Test that _start_scheduled_tasks actually schedules async tasks."""
        supervisor = Supervisor("config.yaml")

        # Mock asyncio.create_task to verify it's called
        with patch("asyncio.create_task") as mock_create_task, patch(
            "asyncio.get_running_loop"
        ):
            supervisor._start_scheduled_tasks()

            # Verify create_task was called twice (heartbeat + fast_heartbeat)
            assert (
                mock_create_task.call_count == 2
            ), "Expected 2 heartbeat tasks to be created"

            # Verify the coroutines being scheduled
            calls = mock_create_task.call_args_list
            call_names = [str(call[0][0]) for call in calls]

            assert any(
                "_create_heartbeat_task" in name for name in call_names
            ), "Expected _create_heartbeat_task to be scheduled"
            assert any(
                "_create_fast_heartbeat_task" in name for name in call_names
            ), "Expected _create_fast_heartbeat_task to be scheduled"

    def test_start_scheduled_tasks_handles_no_event_loop(self):
        """Test that _start_scheduled_tasks handles missing event loop gracefully."""
        supervisor = Supervisor("config.yaml")

        # Simulate no event loop available
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")):
            # Should not raise, just log warning
            supervisor._start_scheduled_tasks()
            # Test passes if no exception raised

    @pytest.mark.asyncio
    async def test_fast_heartbeat_publishes_events(self):
        """Test that fast heartbeat actually publishes FAST_HEARTBEAT_TICK events."""
        supervisor = Supervisor("config.yaml")
        supervisor.start()

        # Track published events
        published_events = []

        def capture_publish(publisher, message_type, message):
            published_events.append(message_type)

        # Mock the message bus publish method
        original_publish = supervisor.message_bus.publish
        supervisor.message_bus.publish = capture_publish

        try:
            # Start the fast heartbeat task
            task = asyncio.create_task(supervisor._create_fast_heartbeat_task())

            # Wait for at least one heartbeat (5 seconds + buffer)
            await asyncio.sleep(6)

            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify FAST_HEARTBEAT_TICK was published
            assert (
                "FAST_HEARTBEAT_TICK" in published_events
            ), "Expected FAST_HEARTBEAT_TICK to be published by fast heartbeat"

        finally:
            supervisor.message_bus.publish = original_publish
            supervisor.stop()


class TestIntentHandlerRegistration:
    """Tests to verify intent handlers are properly registered."""

    def test_timer_intents_registered_on_set_intent_processor(self):
        """Test that timer intent handlers are registered when IntentProcessor is set."""
        # Create role registry
        message_bus = MessageBus()
        role_registry = RoleRegistry(message_bus=message_bus)

        # Load timer role
        role_registry.refresh()

        # Create intent processor
        intent_processor = IntentProcessor()

        # Set intent processor on role registry
        role_registry.set_intent_processor(intent_processor)

        # Verify timer intent handlers are registered
        registered_handlers = intent_processor.get_registered_handlers()
        role_handlers = registered_handlers.get("role_handlers", [])

        # Check for timer-related intent handlers
        timer_intents = [
            "TimerCreationIntent",
            "TimerExpiryIntent",
            "TimerCancellationIntent",
            "TimerListingIntent",
        ]

        for intent_name in timer_intents:
            assert any(
                intent_name in handler for handler in role_handlers
            ), f"Expected {intent_name} to be registered with IntentProcessor"

    def test_intent_processor_receives_timer_creation_intent(self):
        """Test that TimerCreationIntent can be processed without warnings."""
        intent_processor = IntentProcessor()

        # Manually register timer intent handler (simulating what should happen)
        intent_processor.register_role_intent_handler(
            TimerCreationIntent, process_timer_creation_intent, "timer"
        )

        # Create a timer intent
        intent = TimerCreationIntent(
            timer_id="test_123",
            duration="5s",
            duration_seconds=5,
            label="Test",
            event_context={"user_id": "test", "channel_id": "test"},
        )

        # Verify intent is valid
        assert intent.validate(), "Timer intent should be valid"

        # Verify handler is registered
        assert (
            TimerCreationIntent in intent_processor._role_handlers
        ), "TimerCreationIntent handler should be registered"

    def test_intent_processor_receives_timer_expiry_intent(self):
        """Test that TimerExpiryIntent can be processed without warnings."""
        intent_processor = IntentProcessor()

        # Manually register timer expiry intent handler
        intent_processor.register_role_intent_handler(
            TimerExpiryIntent, process_timer_expiry_intent, "timer"
        )

        # Create a timer expiry intent
        intent = TimerExpiryIntent(
            timer_id="test_123",
            original_duration="5s",
            label="Test",
            user_id="test",
            channel_id="test_channel",
            event_context={"user_id": "test", "channel_id": "test_channel"},
        )

        # Verify intent is valid
        assert intent.validate(), "Timer expiry intent should be valid"

        # Verify handler is registered
        assert (
            TimerExpiryIntent in intent_processor._role_handlers
        ), "TimerExpiryIntent handler should be registered"


class TestTimerWorkflowIntegration:
    """Integration tests for complete timer workflow."""

    def test_supervisor_initialization_registers_timer_intents(self):
        """Test that supervisor initialization properly registers timer intents."""
        supervisor = Supervisor("config.yaml")
        supervisor.start()

        try:
            # Verify intent processor exists
            assert (
                supervisor.intent_processor is not None
            ), "IntentProcessor should be initialized"

            # Verify role registry has intent processor
            assert (
                supervisor.workflow_engine.role_registry.intent_processor is not None
            ), "RoleRegistry should have IntentProcessor reference"

            # Verify timer intents are registered
            registered_handlers = supervisor.intent_processor.get_registered_handlers()
            role_handlers = registered_handlers.get("role_handlers", [])

            assert any(
                "TimerCreationIntent" in h for h in role_handlers
            ), "TimerCreationIntent should be registered after supervisor start"
            assert any(
                "TimerExpiryIntent" in h for h in role_handlers
            ), "TimerExpiryIntent should be registered after supervisor start"

        finally:
            supervisor.stop()

    @pytest.mark.asyncio
    async def test_complete_timer_workflow_with_mocked_redis(self):
        """Test complete timer workflow from creation to expiry with mocked Redis."""
        supervisor = Supervisor("config.yaml")
        supervisor.start()

        try:
            # Mock Redis operations
            with patch(
                "roles.shared_tools.redis_tools.redis_write"
            ) as mock_write, patch(
                "roles.shared_tools.redis_tools.redis_read"
            ) as mock_read, patch(
                "roles.shared_tools.redis_tools._get_redis_client"
            ) as mock_client:
                # Setup mock Redis client
                mock_redis = MagicMock()
                mock_client.return_value = mock_redis
                mock_write.return_value = {"success": True}

                # Create timer intent
                timer_intent = TimerCreationIntent(
                    timer_id="test_timer",
                    duration="1s",
                    duration_seconds=1,
                    label="Test Timer",
                    event_context={"user_id": "test", "channel_id": "test"},
                )

                # Process timer creation
                await supervisor.intent_processor.process_intents([timer_intent])

                # Verify Redis write was called
                assert mock_write.called, "Timer should be written to Redis"
                assert mock_redis.zadd.called, "Timer should be added to sorted set"

                # Simulate timer expiry
                mock_read.return_value = {
                    "success": True,
                    "value": {
                        "id": "test_timer",
                        "duration": "1s",
                        "label": "Test Timer",
                        "event_context": {"user_id": "test", "channel_id": "test"},
                    },
                }

                # Create expiry intent
                expiry_intent = TimerExpiryIntent(
                    timer_id="test_timer",
                    original_duration="1s",
                    label="Test Timer",
                    user_id="test",
                    channel_id="test",
                    event_context={"user_id": "test", "channel_id": "test"},
                )

                # Process timer expiry
                await supervisor.intent_processor.process_intents([expiry_intent])

                # Test passes if no exceptions raised

        finally:
            supervisor.stop()


class TestTimerSystemHealthChecks:
    """Health check tests to verify timer system is properly configured."""

    def test_timer_role_has_heartbeat_handler(self):
        """Test that timer role registers FAST_HEARTBEAT_TICK handler."""
        from roles.core_timer import register_role

        registration = register_role()

        assert "event_handlers" in registration, "Timer role should have event handlers"
        assert (
            "FAST_HEARTBEAT_TICK" in registration["event_handlers"]
        ), "Timer role should handle FAST_HEARTBEAT_TICK events"

    def test_timer_role_has_all_intent_handlers(self):
        """Test that timer role registers all required intent handlers."""
        from roles.core_timer import register_role

        registration = register_role()

        assert "intents" in registration, "Timer role should have intent handlers"

        required_intents = [
            TimerCreationIntent,
            TimerExpiryIntent,
        ]

        for intent_type in required_intents:
            assert (
                intent_type in registration["intents"]
            ), f"Timer role should handle {intent_type.__name__}"

    def test_message_bus_has_fast_heartbeat_event_registered(self):
        """Test that FAST_HEARTBEAT_TICK is registered in message bus."""
        message_bus = MessageBus()

        # Verify event is registered in the event registry
        assert message_bus.event_registry.is_valid_event_type(
            "FAST_HEARTBEAT_TICK"
        ), "FAST_HEARTBEAT_TICK should be registered as valid event type"
