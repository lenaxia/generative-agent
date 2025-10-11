"""
Integration tests for Timer Role Event-Driven Workflows.

Tests the end-to-end functionality of timer events, including event publishing,
handler execution, and workflow creation from timer expiry events.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.event_handler_llm import EventHandlerLLM
from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry
from roles.timer.lifecycle import (
    handle_location_based_timer_update,
    handle_timer_expiry_action,
)


class TestTimerEventWorkflows:
    """Integration tests for timer event-driven workflows."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        mock_model = AsyncMock()
        mock_model.invoke = AsyncMock()
        factory.create_strands_model = Mock(return_value=mock_model)
        return factory

    @pytest.fixture
    def mock_workflow_engine(self):
        """Create a mock workflow engine."""
        engine = AsyncMock()
        engine.start_workflow = AsyncMock(return_value="workflow_123")
        return engine

    @pytest.fixture
    def mock_communication_manager(self):
        """Create a mock communication manager."""
        comm_mgr = AsyncMock()
        comm_mgr.send_notification = AsyncMock()
        return comm_mgr

    @pytest.fixture
    def timer_expiry_event_data(self):
        """Sample timer expiry event data."""
        return {
            "timer_id": "timer_abc123",
            "original_request": "turn on the lights",
            "execution_context": {
                "user_id": "U123456",
                "channel": "#general",
                "device_context": {
                    "room": "bedroom",
                    "device_id": "echo_dot_bedroom",
                    "available_devices": ["bedroom_lights", "bedside_lamp"],
                },
                "timestamp": "2025-01-01T22:30:00Z",
                "source": "slack",
            },
        }

    @pytest.mark.asyncio
    async def test_timer_expiry_creates_workflow(
        self,
        mock_llm_factory,
        mock_workflow_engine,
        mock_communication_manager,
        timer_expiry_event_data,
    ):
        """Test that timer expiry creates a workflow for action requests."""
        # Mock LLM responses
        mock_model = mock_llm_factory.create_strands_model.return_value
        mock_model.invoke.side_effect = [
            "A) Create a workflow",  # Decision response
            "turn on bedroom lights to 50% brightness",  # Workflow instruction
        ]

        # Create EventHandlerContext
        from common.event_handler_context import EventHandlerContext

        llm = EventHandlerLLM(mock_llm_factory, timer_expiry_event_data)
        ctx = EventHandlerContext(
            llm=llm,
            workflow_engine=mock_workflow_engine,
            communication_manager=mock_communication_manager,
            message_bus=Mock(),
            execution_context=timer_expiry_event_data["execution_context"],
        )

        # Ensure the workflow_engine mock is properly configured
        mock_workflow_engine.start_workflow.return_value = "workflow_123"

        # Call the handler
        await handle_timer_expiry_action(timer_expiry_event_data, ctx)

        # Should create workflow, not send notification
        mock_workflow_engine.start_workflow.assert_called_once()
        mock_communication_manager.send_notification.assert_not_called()

        # Check workflow creation parameters
        call_args = mock_workflow_engine.start_workflow.call_args
        assert "turn on bedroom lights" in call_args.kwargs["instruction"]
        assert (
            call_args.kwargs["context"] == timer_expiry_event_data["execution_context"]
        )

    @pytest.mark.asyncio
    async def test_timer_expiry_sends_notification(
        self,
        mock_llm_factory,
        mock_workflow_engine,
        mock_communication_manager,
        timer_expiry_event_data,
    ):
        """Test that timer expiry sends notification for reminder requests."""
        # Modify event data for reminder
        reminder_event_data = timer_expiry_event_data.copy()
        reminder_event_data["original_request"] = "remind me to check email"

        # Mock LLM responses
        mock_model = mock_llm_factory.create_strands_model.return_value
        mock_model.invoke.return_value = "B) Send a notification"  # Decision response

        # Create EventHandlerContext
        from common.event_handler_context import EventHandlerContext

        llm = EventHandlerLLM(mock_llm_factory, reminder_event_data)
        ctx = EventHandlerContext(
            llm=llm,
            workflow_engine=mock_workflow_engine,
            communication_manager=mock_communication_manager,
            message_bus=Mock(),
            execution_context=reminder_event_data["execution_context"],
        )

        # Call the handler
        await handle_timer_expiry_action(reminder_event_data, ctx)

        # Should send notification, not create workflow
        mock_communication_manager.send_notification.assert_called_once()
        mock_workflow_engine.start_workflow.assert_not_called()

        # Check notification parameters
        call_args = mock_communication_manager.send_notification.call_args
        assert "remind me to check email" in call_args.kwargs["message"]
        assert call_args.kwargs["recipient"] == "#general"

    @pytest.mark.asyncio
    async def test_timer_expiry_fallback_on_error(
        self,
        mock_llm_factory,
        mock_workflow_engine,
        mock_communication_manager,
        timer_expiry_event_data,
    ):
        """Test that timer expiry falls back to notification on errors."""
        # Mock LLM to raise an error
        mock_model = mock_llm_factory.create_strands_model.return_value
        mock_model.invoke.side_effect = Exception("LLM error")

        # Create EventHandlerContext
        from common.event_handler_context import EventHandlerContext

        llm = EventHandlerLLM(mock_llm_factory, timer_expiry_event_data)
        ctx = EventHandlerContext(
            llm=llm,
            workflow_engine=mock_workflow_engine,
            communication_manager=mock_communication_manager,
            message_bus=Mock(),
            execution_context=timer_expiry_event_data["execution_context"],
        )

        # Call the handler
        await handle_timer_expiry_action(timer_expiry_event_data, ctx)

        # Should fallback to notification
        mock_communication_manager.send_notification.assert_called_once()
        mock_workflow_engine.start_workflow.assert_not_called()

        # Check fallback notification
        call_args = mock_communication_manager.send_notification.call_args
        assert "Timer reminder: turn on the lights" in call_args.kwargs["message"]

    @pytest.mark.asyncio
    async def test_timer_expiry_workflow_creation_failure(
        self,
        mock_llm_factory,
        mock_workflow_engine,
        mock_communication_manager,
        timer_expiry_event_data,
    ):
        """Test handling when workflow creation fails."""
        # Mock LLM responses
        mock_model = mock_llm_factory.create_strands_model.return_value
        mock_model.invoke.side_effect = [
            "A) Create a workflow",  # Decision response
            "turn on bedroom lights",  # Workflow instruction
        ]

        # Mock workflow engine to fail
        mock_workflow_engine.start_workflow.side_effect = Exception(
            "Workflow creation failed"
        )

        # Create EventHandlerContext
        from common.event_handler_context import EventHandlerContext

        llm = EventHandlerLLM(mock_llm_factory, timer_expiry_event_data)
        ctx = EventHandlerContext(
            llm=llm,
            workflow_engine=mock_workflow_engine,
            communication_manager=mock_communication_manager,
            message_bus=Mock(),
            execution_context=timer_expiry_event_data["execution_context"],
        )

        # Call the handler
        await handle_timer_expiry_action(timer_expiry_event_data, ctx)

        # Should attempt workflow creation
        mock_workflow_engine.start_workflow.assert_called_once()

        # Should not send fallback notification in this case (handler doesn't implement this yet)
        # This test documents current behavior

    @pytest.mark.asyncio
    async def test_location_based_timer_update_handler(
        self, mock_llm_factory, mock_workflow_engine, mock_communication_manager
    ):
        """Test location-based timer update handler."""
        location_event_data = {
            "user_id": "U123456",
            "location": {"room": "kitchen", "device_id": "echo_kitchen"},
            "affects_timers": True,
            "execution_context": {"user_id": "U123456", "source": "location_service"},
        }

        # Create EventHandlerContext
        from common.event_handler_context import EventHandlerContext

        llm = EventHandlerLLM(mock_llm_factory, location_event_data)
        ctx = EventHandlerContext(
            llm=llm,
            workflow_engine=mock_workflow_engine,
            communication_manager=mock_communication_manager,
            message_bus=Mock(),
            execution_context=location_event_data["execution_context"],
        )

        # Call the handler (should not raise errors)
        await handle_location_based_timer_update(location_event_data, ctx)

        # Currently just logs, so no assertions on external calls
        # This test ensures the handler doesn't crash

    @pytest.mark.asyncio
    async def test_event_handler_llm_integration(
        self, mock_llm_factory, timer_expiry_event_data
    ):
        """Test EventHandlerLLM integration in timer handlers."""
        # Mock LLM responses
        mock_model = mock_llm_factory.create_strands_model.return_value
        mock_model.invoke.return_value = "Test LLM response"

        # Create EventHandlerLLM
        llm = EventHandlerLLM(mock_llm_factory, timer_expiry_event_data)

        # Test convenience methods
        assert llm.get_original_request() == "turn on the lights"
        assert llm.get_timer_id() == "timer_abc123"
        assert llm.get_user_id() == "U123456"
        assert llm.get_channel() == "#general"
        assert llm.get_context("device_context.room") == "bedroom"

        # Test LLM invocation
        response = await llm.invoke("Test prompt")
        assert response == "Test LLM response"

        # Verify context was merged into prompt
        call_args = mock_model.invoke.call_args[0][0]
        assert "user_id" in call_args
        assert "U123456" in call_args

    def test_timer_role_event_registration_integration(self):
        """Test that timer role events are properly registered during role loading."""
        # Create MessageBus and RoleRegistry
        message_bus = MessageBus()
        role_registry = RoleRegistry("roles", message_bus=message_bus)

        # Load timer role
        timer_role = role_registry.get_role("timer")

        # Verify role has events
        assert "events" in timer_role.config
        events = timer_role.config["events"]

        # Check published events
        published_events = [e["event_type"] for e in events.get("publishes", [])]
        assert "TIMER_EXPIRED" in published_events
        assert "TIMER_CREATED" in published_events

        # Check subscribed events
        subscribed_events = [e["event_type"] for e in events.get("subscribes", [])]
        assert "TIMER_EXPIRED" in subscribed_events
        assert "USER_LOCATION_CHANGED" in subscribed_events

        # Check event registry
        event_docs = message_bus.event_registry.get_event_documentation()
        assert "TIMER_EXPIRED" in event_docs["registered_events"]
        assert "timer" in event_docs["publishers"]["TIMER_EXPIRED"]
        assert "timer" in event_docs["subscribers"]["TIMER_EXPIRED"]

    @pytest.mark.asyncio
    async def test_end_to_end_timer_event_flow(self):
        """Test complete timer event flow from registration to handler execution."""
        # Create real MessageBus and register timer events
        message_bus = MessageBus()
        message_bus.start()

        # Mock dependencies for handler
        mock_llm_factory = Mock(spec=LLMFactory)
        mock_model = AsyncMock()
        mock_model.invoke.return_value = "B) Send a notification"
        mock_llm_factory.create_strands_model.return_value = mock_model

        mock_workflow_engine = AsyncMock()
        mock_communication_manager = AsyncMock()

        # Inject dependencies into MessageBus
        message_bus.llm_factory = mock_llm_factory
        message_bus.workflow_engine = mock_workflow_engine
        message_bus.communication_manager = mock_communication_manager

        # Load timer role with event registration
        role_registry = RoleRegistry("roles", message_bus=message_bus)
        timer_role = role_registry.get_role("timer")

        # Verify handler was registered
        assert "timer" in message_bus._subscribers.get("TIMER_EXPIRED", {})

        # Publish TIMER_EXPIRED event
        timer_event_data = {
            "timer_id": "test123",
            "original_request": "remind me to check email",
            "execution_context": {
                "user_id": "U123456",
                "channel": "#general",
                "source": "slack",
            },
        }

        message_bus.publish("timer", "TIMER_EXPIRED", timer_event_data)

        # Wait for async handler execution
        await asyncio.sleep(0.2)

        # Verify handler was called (notification should be sent)
        mock_communication_manager.send_notification.assert_called_once()
        call_args = mock_communication_manager.send_notification.call_args
        assert "remind me to check email" in call_args.kwargs["message"]
