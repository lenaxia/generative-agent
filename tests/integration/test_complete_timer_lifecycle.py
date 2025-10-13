"""
Integration test for complete timer lifecycle with new event-driven architecture.

Tests the end-to-end flow: Timer creation → FastHeartbeat monitoring → Timer expiry →
Event publishing → Handler execution → Workflow creation.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.event_handler_context import EventHandlerContext
from common.message_bus import MessageBus
from llm_provider.role_registry import RoleRegistry

# REMOVED: from supervisor.fast_heartbeat import FastHeartbeat - using scheduled tasks now
from supervisor.supervisor import Supervisor


class TestCompleteTimerLifecycle:
    """Test complete timer lifecycle with new event-driven architecture."""

    @pytest.mark.asyncio
    async def test_fast_heartbeat_publishes_events(self):
        """Test that FastHeartbeat publishes FAST_HEARTBEAT_TICK events."""
        message_bus = MessageBus()
        message_bus.start()

        # Create FastHeartbeat with very short interval for testing
        # REMOVED: FastHeartbeat - using scheduled tasks now
        # fast_heartbeat = FastHeartbeat(message_bus, interval=0.1)

        # Track published events
        published_events = []

        def capture_event(event_data):
            published_events.append(event_data)

        # Subscribe to FAST_HEARTBEAT_TICK events
        message_bus.subscribe("test_subscriber", "FAST_HEARTBEAT_TICK", capture_event)

        # Start FastHeartbeat
        # REMOVED: fast_heartbeat.start() - using scheduled tasks now

        # Wait for at least one heartbeat
        await asyncio.sleep(0.3)

        # Stop FastHeartbeat
        fast_heartbeat.stop()

        # Verify events were published
        assert len(published_events) >= 1

        # Verify event format
        event = published_events[0]
        assert "timestamp" in event
        assert "interval" in event
        assert event["interval"] == 0.1

    @pytest.mark.asyncio
    async def test_timer_role_heartbeat_subscription_integration(self):
        """Test that timer role properly subscribes to FAST_HEARTBEAT_TICK events."""
        message_bus = MessageBus()
        message_bus.start()

        # Mock dependencies for handler
        mock_llm_factory = Mock()
        mock_workflow_engine = AsyncMock()
        mock_communication_manager = AsyncMock()

        # Inject dependencies into MessageBus
        message_bus.llm_factory = mock_llm_factory
        message_bus.workflow_engine = mock_workflow_engine
        message_bus.communication_manager = mock_communication_manager

        # Load timer role with event registration
        role_registry = RoleRegistry("roles", message_bus=message_bus)
        timer_role = role_registry.get_role("timer")

        # Verify timer role subscribed to FAST_HEARTBEAT_TICK
        assert "FAST_HEARTBEAT_TICK" in message_bus._subscribers
        assert "timer" in message_bus._subscribers["FAST_HEARTBEAT_TICK"]

        # Mock timer manager to return no expired timers
        with patch("roles.timer.lifecycle.get_timer_manager") as mock_get_timer_manager:
            mock_timer_manager = AsyncMock()
            mock_timer_manager.get_expiring_timers.return_value = []
            mock_get_timer_manager.return_value = mock_timer_manager

            # Publish FAST_HEARTBEAT_TICK event
            heartbeat_data = {
                "timestamp": datetime.now().isoformat(),
                "interval": 5,
                "tick_count": 1,
            }

            message_bus.publish("fast_heartbeat", "FAST_HEARTBEAT_TICK", heartbeat_data)

            # Wait for async handler execution
            await asyncio.sleep(0.1)

            # Verify timer manager was called to check for expired timers
            mock_timer_manager.get_expiring_timers.assert_called_once()

    @pytest.mark.asyncio
    async def test_timer_expiry_event_publishing_integration(self):
        """Test that expired timers publish TIMER_EXPIRED events in correct format."""
        message_bus = MessageBus()
        message_bus.start()

        # Mock dependencies
        mock_llm_factory = Mock()
        mock_workflow_engine = AsyncMock()
        mock_communication_manager = AsyncMock()

        message_bus.llm_factory = mock_llm_factory
        message_bus.workflow_engine = mock_workflow_engine
        message_bus.communication_manager = mock_communication_manager

        # Load timer role
        role_registry = RoleRegistry("roles", message_bus=message_bus)
        timer_role = role_registry.get_role("timer")

        # Track TIMER_EXPIRED events
        timer_expired_events = []

        def capture_timer_expired(event_data):
            timer_expired_events.append(event_data)

        message_bus.subscribe("test_subscriber", "TIMER_EXPIRED", capture_timer_expired)

        # Mock expired timer with rich context
        expired_timer = {
            "id": "timer_123",
            "name": "Test Timer",
            "user_id": "U123456",
            "channel_id": "slack:#general",
            "request_context": {
                "original_request": "turn on the lights",
                "execution_context": {
                    "user_id": "U123456",
                    "channel": "#general",
                    "device_context": {"room": "bedroom"},
                    "timestamp": "2025-01-01T22:30:00Z",
                    "source": "slack",
                },
            },
        }

        # Mock timer manager to return expired timer
        with patch("roles.timer.lifecycle.get_timer_manager") as mock_get_timer_manager:
            mock_timer_manager = AsyncMock()
            mock_timer_manager.get_expiring_timers.return_value = [expired_timer]
            mock_timer_manager.update_timer_status = AsyncMock()
            mock_get_timer_manager.return_value = mock_timer_manager

            # Publish FAST_HEARTBEAT_TICK to trigger timer monitoring
            heartbeat_data = {"timestamp": datetime.now().isoformat(), "interval": 5}

            message_bus.publish("fast_heartbeat", "FAST_HEARTBEAT_TICK", heartbeat_data)

            # Wait for async processing
            await asyncio.sleep(0.2)

            # Verify TIMER_EXPIRED event was published in correct format
            assert len(timer_expired_events) == 1

            timer_event = timer_expired_events[0]
            assert timer_event["timer_id"] == "timer_123"
            assert timer_event["original_request"] == "turn on the lights"
            assert "execution_context" in timer_event
            assert timer_event["execution_context"]["user_id"] == "U123456"
            assert timer_event["execution_context"]["channel"] == "#general"

    def test_event_system_documentation(self):
        """Test that event system provides complete documentation."""
        message_bus = MessageBus()
        role_registry = RoleRegistry("roles", message_bus=message_bus)

        # Get complete event documentation
        event_docs = message_bus.event_registry.get_event_documentation()

        # Should have all system events
        expected_system_events = [
            "WORKFLOW_STARTED",
            "WORKFLOW_COMPLETED",
            "WORKFLOW_FAILED",
            "TASK_STARTED",
            "TASK_COMPLETED",
            "TASK_FAILED",
            "AGENT_ROLE_SWITCHED",
            "SYSTEM_HEALTH_CHECK",
            "HEARTBEAT_TICK",
            "FAST_HEARTBEAT_TICK",
        ]

        for event in expected_system_events:
            assert (
                event in event_docs["registered_events"]
            ), f"Missing system event: {event}"

        # Should have timer events
        assert "TIMER_EXPIRED" in event_docs["registered_events"]
        assert "TIMER_CREATED" in event_docs["registered_events"]

        # Should have proper publisher/subscriber mappings
        assert "system" in event_docs["publishers"]["FAST_HEARTBEAT_TICK"]
        assert "timer" in event_docs["publishers"]["TIMER_EXPIRED"]
        assert "timer" in event_docs["subscribers"]["FAST_HEARTBEAT_TICK"]
        assert "timer" in event_docs["subscribers"]["TIMER_EXPIRED"]

    def test_backward_compatibility_maintained(self):
        """Test that old MessageType enum still works alongside new dynamic events."""
        from common.message_bus import MessageType

        message_bus = MessageBus()
        message_bus.start()

        # Test old MessageType enum still works
        callback_called = []

        def old_style_callback(data):
            callback_called.append(data)

        # Subscribe using old MessageType enum
        message_bus.subscribe("test", MessageType.TIMER_EXPIRED, old_style_callback)

        # Publish using old MessageType enum
        message_bus.publish("test", MessageType.TIMER_EXPIRED, {"test": "data"})

        # Wait for callback
        time.sleep(0.1)

        # Should work with backward compatibility
        assert len(callback_called) == 1
        assert callback_called[0]["test"] == "data"
