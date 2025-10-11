"""
Unit tests for Timer Monitoring Architecture Refactor.

Tests the EventHandlerContext pattern, dual heartbeat system, and timer role self-monitoring.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.event_handler_context import EventHandlerContext
from common.event_handler_llm import EventHandlerLLM
from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory
from supervisor.fast_heartbeat import FastHeartbeat


class TestEventHandlerContext:
    """Test EventHandlerContext pattern for clean dependency injection."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for EventHandlerContext."""
        return {
            "llm": Mock(spec=EventHandlerLLM),
            "workflow_engine": AsyncMock(),
            "communication_manager": AsyncMock(),
            "message_bus": Mock(spec=MessageBus),
            "execution_context": {"user_id": "U123", "channel": "#general"},
        }

    def test_event_handler_context_creation(self, mock_dependencies):
        """Test EventHandlerContext creation with all dependencies."""
        ctx = EventHandlerContext(**mock_dependencies)

        assert ctx.llm == mock_dependencies["llm"]
        assert ctx.workflow_engine == mock_dependencies["workflow_engine"]
        assert ctx.communication_manager == mock_dependencies["communication_manager"]
        assert ctx.message_bus == mock_dependencies["message_bus"]
        assert ctx.execution_context == mock_dependencies["execution_context"]

    def test_event_handler_context_convenience_methods(self, mock_dependencies):
        """Test convenience methods on EventHandlerContext."""
        ctx = EventHandlerContext(**mock_dependencies)

        # Test delegation to llm for common operations
        ctx.llm.get_user_id.return_value = "U123456"
        ctx.llm.get_channel.return_value = "#general"
        ctx.llm.get_original_request.return_value = "turn on lights"

        assert ctx.get_user_id() == "U123456"
        assert ctx.get_channel() == "#general"
        assert ctx.get_original_request() == "turn on lights"

    @pytest.mark.asyncio
    async def test_event_handler_context_workflow_creation(self, mock_dependencies):
        """Test workflow creation through context."""
        ctx = EventHandlerContext(**mock_dependencies)
        ctx.workflow_engine.start_workflow.return_value = "workflow_123"

        workflow_id = await ctx.create_workflow("turn on bedroom lights")

        ctx.workflow_engine.start_workflow.assert_called_once_with(
            instruction="turn on bedroom lights", context=ctx.execution_context
        )
        assert workflow_id == "workflow_123"

    @pytest.mark.asyncio
    async def test_event_handler_context_notification_sending(self, mock_dependencies):
        """Test notification sending through context."""
        ctx = EventHandlerContext(**mock_dependencies)

        await ctx.send_notification("Test message")

        ctx.communication_manager.send_notification.assert_called_once()

    def test_event_handler_context_event_publishing(self, mock_dependencies):
        """Test event publishing through context."""
        ctx = EventHandlerContext(**mock_dependencies)

        event_data = {"timer_id": "test123", "action": "completed"}
        ctx.publish_event("TIMER_COMPLETED", event_data)

        ctx.message_bus.publish.assert_called_once_with(
            ctx, "TIMER_COMPLETED", event_data
        )


class TestFastHeartbeat:
    """Test FastHeartbeat service for high-frequency monitoring."""

    @pytest.fixture
    def mock_message_bus(self):
        """Create mock MessageBus."""
        bus = Mock(spec=MessageBus)
        bus.publish = Mock()
        return bus

    def test_fast_heartbeat_initialization(self, mock_message_bus):
        """Test FastHeartbeat initialization."""
        fast_heartbeat = FastHeartbeat(mock_message_bus, interval=5)

        assert fast_heartbeat.message_bus == mock_message_bus
        assert fast_heartbeat.interval == 5
        assert fast_heartbeat.stop_event is not None
        assert not fast_heartbeat.is_running

    def test_fast_heartbeat_start_stop(self, mock_message_bus):
        """Test FastHeartbeat start and stop functionality."""
        fast_heartbeat = FastHeartbeat(mock_message_bus, interval=5)

        # Test start
        fast_heartbeat.start()
        assert fast_heartbeat.is_running
        assert fast_heartbeat.thread.is_alive()

        # Test stop
        fast_heartbeat.stop()
        fast_heartbeat.thread.join(timeout=1)  # Wait for thread to stop
        assert not fast_heartbeat.is_running

    def test_fast_heartbeat_event_publishing(self, mock_message_bus):
        """Test that FastHeartbeat publishes FAST_HEARTBEAT_TICK events."""
        fast_heartbeat = FastHeartbeat(
            mock_message_bus, interval=0.1
        )  # Very fast for testing

        fast_heartbeat.start()

        # Wait for at least one heartbeat
        import time

        time.sleep(0.2)

        fast_heartbeat.stop()
        fast_heartbeat.thread.join(timeout=1)

        # Should have published at least one FAST_HEARTBEAT_TICK event
        mock_message_bus.publish.assert_called()
        call_args = mock_message_bus.publish.call_args_list
        assert any("FAST_HEARTBEAT_TICK" in str(call) for call in call_args)


class TestTimerRoleSelfMonitoring:
    """Test timer role self-monitoring with heartbeat subscription."""

    @pytest.fixture
    def mock_timer_manager(self):
        """Create mock TimerManager."""
        manager = AsyncMock()
        manager.get_expiring_timers.return_value = []
        manager.update_timer_status = AsyncMock()
        return manager

    @pytest.fixture
    def mock_context(self):
        """Create real EventHandlerContext with mocked dependencies."""
        from common.event_handler_context import EventHandlerContext
        from common.event_handler_llm import EventHandlerLLM

        mock_message_bus = Mock()
        mock_message_bus.publish = Mock()

        mock_llm = Mock(spec=EventHandlerLLM)

        ctx = EventHandlerContext(
            llm=mock_llm,
            workflow_engine=AsyncMock(),
            communication_manager=AsyncMock(),
            message_bus=mock_message_bus,
            execution_context={},
        )
        return ctx

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring_handler_no_expired_timers(
        self, mock_timer_manager, mock_context
    ):
        """Test heartbeat monitoring when no timers are expired."""
        from roles.timer.lifecycle import handle_heartbeat_monitoring

        with patch(
            "roles.timer.lifecycle.get_timer_manager", return_value=mock_timer_manager
        ):
            # Mock no expired timers
            mock_timer_manager.get_expiring_timers.return_value = []

            event_data = {"timestamp": "2025-01-01T12:00:00Z", "interval": 5}

            # Should not raise errors
            await handle_heartbeat_monitoring(event_data, mock_context)

            # Should check for expired timers
            mock_timer_manager.get_expiring_timers.assert_called_once()

            # Should not publish any TIMER_EXPIRED events
            mock_context.message_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring_handler_with_expired_timers(
        self, mock_timer_manager, mock_context
    ):
        """Test heartbeat monitoring when timers are expired."""
        from roles.timer.lifecycle import handle_heartbeat_monitoring

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

        with patch(
            "roles.timer.lifecycle.get_timer_manager", return_value=mock_timer_manager
        ):
            mock_timer_manager.get_expiring_timers.return_value = [expired_timer]

            event_data = {"timestamp": "2025-01-01T12:00:00Z", "interval": 5}

            await handle_heartbeat_monitoring(event_data, mock_context)

            # Should publish TIMER_EXPIRED event in new format
            mock_context.message_bus.publish.assert_called_once()
            call_args = mock_context.message_bus.publish.call_args

            # Verify event type and data format
            assert call_args[0][1] == "TIMER_EXPIRED"  # event_type
            published_data = call_args[0][2]  # event_data

            assert published_data["timer_id"] == "timer_123"
            assert published_data["original_request"] == "turn on the lights"
            assert "execution_context" in published_data
            assert published_data["execution_context"]["user_id"] == "U123456"

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring_handler_error_handling(
        self, mock_timer_manager, mock_context
    ):
        """Test error handling in heartbeat monitoring."""
        from roles.timer.lifecycle import handle_heartbeat_monitoring

        with patch(
            "roles.timer.lifecycle.get_timer_manager", return_value=mock_timer_manager
        ):
            # Mock timer manager to raise error
            mock_timer_manager.get_expiring_timers.side_effect = Exception(
                "Redis error"
            )

            event_data = {"timestamp": "2025-01-01T12:00:00Z", "interval": 5}

            # Should not raise error (should handle gracefully)
            await handle_heartbeat_monitoring(event_data, mock_context)

            # Should not publish events on error
            mock_context.message_bus.publish.assert_not_called()

    def test_timer_role_heartbeat_subscription_declaration(self):
        """Test that timer role declares FAST_HEARTBEAT_TICK subscription."""
        import yaml

        with open("roles/timer/definition.yaml") as f:
            timer_config = yaml.safe_load(f)

        # Should have events section
        assert "events" in timer_config

        # Should subscribe to FAST_HEARTBEAT_TICK
        subscribes = timer_config["events"].get("subscribes", [])
        heartbeat_subscriptions = [
            sub for sub in subscribes if sub["event_type"] == "FAST_HEARTBEAT_TICK"
        ]

        assert len(heartbeat_subscriptions) == 1
        assert heartbeat_subscriptions[0]["handler"] == "handle_heartbeat_monitoring"
