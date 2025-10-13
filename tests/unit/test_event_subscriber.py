"""
Unit tests for the EventSubscriber class.

This test suite covers the EventSubscriber class in roles/shared_tools/event_subscriber.py,
testing event subscription, event handling, and action execution for both happy and
error paths.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.message_bus import MessageBus, MessageType
from roles.shared_tools.event_subscriber import ActionType, EventSubscriber


class TestEventSubscriber:
    """Tests for the EventSubscriber class."""

    def test_initialization(self):
        """Test initializing the event subscriber."""
        workflow_engine = Mock()
        universal_agent = Mock()
        message_bus = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
            message_bus=message_bus,
        )

        assert subscriber.role_name == "timer"
        assert subscriber.workflow_engine == workflow_engine
        assert subscriber.universal_agent == universal_agent
        assert subscriber.message_bus == message_bus
        assert MessageType.TIMER_EXPIRED in subscriber.event_handlers

    def test_default_handlers_registration(self):
        """Test default event handlers are properly registered."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Check default handlers are registered
        assert MessageType.TIMER_EXPIRED in subscriber.event_handlers
        assert (
            subscriber.event_handlers[MessageType.TIMER_EXPIRED]
            == subscriber.handle_delayed_action
        )

    @pytest.mark.asyncio
    async def test_subscribe_to_events(self):
        """Test subscribing to events on the message bus."""
        workflow_engine = Mock()
        universal_agent = Mock()
        message_bus = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Add custom handler
        custom_handler = AsyncMock()
        subscriber.add_custom_handler(MessageType.WORKFLOW_COMPLETED, custom_handler)

        # Subscribe to events
        await subscriber.subscribe_to_events(message_bus)

        # Verify subscriptions
        expected_calls = [
            (
                (
                    subscriber,
                    MessageType.TIMER_EXPIRED,
                    subscriber.handle_delayed_action,
                ),
                {},
            ),
            ((subscriber, MessageType.WORKFLOW_COMPLETED, custom_handler), {}),
        ]

        assert message_bus.subscribe.call_count == 2
        assert message_bus.subscribe.call_args_list[0][0][0] == subscriber
        assert (
            message_bus.subscribe.call_args_list[0][0][1] == MessageType.TIMER_EXPIRED
        )
        assert message_bus.subscribe.call_args_list[1][0][0] == subscriber
        assert (
            message_bus.subscribe.call_args_list[1][0][1]
            == MessageType.WORKFLOW_COMPLETED
        )

    def test_add_custom_handler(self):
        """Test adding a custom event handler."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Add custom handler
        custom_handler = AsyncMock()
        subscriber.add_custom_handler(MessageType.WORKFLOW_COMPLETED, custom_handler)

        # Verify handler was added
        assert MessageType.WORKFLOW_COMPLETED in subscriber.event_handlers
        assert (
            subscriber.event_handlers[MessageType.WORKFLOW_COMPLETED] == custom_handler
        )

    @pytest.mark.asyncio
    async def test_parse_delayed_action_success(self):
        """Test successful parsing of delayed action."""
        workflow_engine = Mock()
        universal_agent = Mock()
        router_mock = Mock()

        # Mock response from universal agent
        parse_response = json.dumps(
            {
                "action_type": "notification",
                "workflow_instruction": "Send notification about timer expiry",
                "target_context": {"channel": "slack"},
                "fallback_notification": "Timer reminder",
            }
        )

        universal_agent.assume_role.return_value = router_mock
        router_mock.execute_task = AsyncMock(return_value=parse_response)

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {
                "user_id": "user123",
                "channel": "slack",
                "timestamp": "2023-01-01T12:00:00Z",
            },
        }

        # Execute parsing
        result = await subscriber._parse_delayed_action(event_data)

        # Verify universal agent was called with router role
        universal_agent.assume_role.assert_called_once_with("router")
        router_mock.execute_task.assert_called_once()
        assert (
            "Parse this delayed action request"
            in router_mock.execute_task.call_args[0][0]
        )

        # Verify parsing result
        assert result["action_type"] == "notification"
        assert result["workflow_instruction"] == "Send notification about timer expiry"
        assert result["target_context"]["channel"] == "slack"
        assert result["fallback_notification"] == "Timer reminder"

    @pytest.mark.asyncio
    async def test_parse_delayed_action_error(self):
        """Test error handling in parsing delayed action."""
        workflow_engine = Mock()
        universal_agent = Mock()
        router_mock = Mock()

        # Mock execution error
        universal_agent.assume_role.return_value = router_mock
        router_mock.execute_task = AsyncMock(side_effect=Exception("Parsing failed"))

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123", "channel": "slack"},
        }

        # Execute parsing
        result = await subscriber._parse_delayed_action(event_data)

        # Verify fallback response on error
        assert result["action_type"] == "notification"
        assert "Reminder: Remind me in 5 minutes" in result["workflow_instruction"]
        assert (
            "Timer reminder: Remind me in 5 minutes" in result["fallback_notification"]
        )

    @pytest.mark.asyncio
    async def test_handle_delayed_action_notification_type(self):
        """Test handling delayed action of notification type."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Mock _parse_delayed_action to return notification action
        parsed_action = {
            "action_type": ActionType.NOTIFICATION.value,
            "workflow_instruction": "Timer expired notification",
            "target_context": {"channel": "slack"},
            "fallback_notification": "Timer reminder",
        }
        subscriber._parse_delayed_action = AsyncMock(return_value=parsed_action)
        subscriber._handle_notification_action = AsyncMock()

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber.handle_delayed_action(event_data)

        # Verify notification handling was called
        subscriber._parse_delayed_action.assert_called_once_with(event_data)
        subscriber._handle_notification_action.assert_called_once_with(
            parsed_action, event_data
        )

    @pytest.mark.asyncio
    async def test_handle_delayed_action_workflow_type(self):
        """Test handling delayed action of workflow type."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Mock _parse_delayed_action to return workflow action
        parsed_action = {
            "action_type": ActionType.WORKFLOW.value,
            "workflow_instruction": "Create workflow for timer expiry",
            "target_context": {"channel": "slack"},
            "fallback_notification": "Timer reminder",
        }
        subscriber._parse_delayed_action = AsyncMock(return_value=parsed_action)
        subscriber._handle_workflow_action = AsyncMock()

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber.handle_delayed_action(event_data)

        # Verify workflow handling was called
        subscriber._parse_delayed_action.assert_called_once_with(event_data)
        subscriber._handle_workflow_action.assert_called_once_with(
            parsed_action, event_data
        )

    @pytest.mark.asyncio
    async def test_handle_delayed_action_direct_action_type(self):
        """Test handling delayed action of direct action type."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Mock _parse_delayed_action to return direct action
        parsed_action = {
            "action_type": ActionType.DIRECT_ACTION.value,
            "workflow_instruction": "Execute direct action",
            "target_context": {"channel": "slack"},
            "fallback_notification": "Timer reminder",
        }
        subscriber._parse_delayed_action = AsyncMock(return_value=parsed_action)
        subscriber._handle_direct_action = AsyncMock()

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber.handle_delayed_action(event_data)

        # Verify direct action handling was called
        subscriber._parse_delayed_action.assert_called_once_with(event_data)
        subscriber._handle_direct_action.assert_called_once_with(
            parsed_action, event_data
        )

    @pytest.mark.asyncio
    async def test_handle_delayed_action_unknown_type(self):
        """Test handling delayed action with unknown type."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Mock _parse_delayed_action to return unknown action type
        parsed_action = {
            "action_type": "unknown_type",
            "workflow_instruction": "Unknown action",
            "target_context": {"channel": "slack"},
            "fallback_notification": "Timer reminder",
        }
        subscriber._parse_delayed_action = AsyncMock(return_value=parsed_action)
        subscriber._handle_notification_fallback = AsyncMock()

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber.handle_delayed_action(event_data)

        # Verify fallback was called for unknown type
        subscriber._parse_delayed_action.assert_called_once_with(event_data)
        subscriber._handle_notification_fallback.assert_called_once_with(event_data)

    @pytest.mark.asyncio
    async def test_handle_delayed_action_error(self):
        """Test error handling in delayed action processing."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Mock _parse_delayed_action to throw error
        subscriber._parse_delayed_action = AsyncMock(
            side_effect=Exception("Parsing error")
        )
        subscriber._handle_notification_fallback = AsyncMock()

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber.handle_delayed_action(event_data)

        # Verify fallback was called on error
        subscriber._parse_delayed_action.assert_called_once_with(event_data)
        subscriber._handle_notification_fallback.assert_called_once_with(event_data)

    @pytest.mark.asyncio
    async def test_handle_workflow_action(self):
        """Test handling workflow action."""
        workflow_engine = Mock()
        universal_agent = Mock()

        # Mock workflow creation
        workflow_engine.start_workflow = AsyncMock(return_value="workflow-123")

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        parsed_action = {"workflow_instruction": "Create summary report"}

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber._handle_workflow_action(parsed_action, event_data)

        # Verify workflow was created
        workflow_engine.start_workflow.assert_called_once_with(
            instruction="Create summary report", context=event_data["execution_context"]
        )

    @pytest.mark.asyncio
    async def test_handle_workflow_action_missing_instruction(self):
        """Test handling workflow action with missing instruction."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        # Empty parsed action (no workflow_instruction)
        parsed_action = {}
        subscriber._handle_notification_fallback = AsyncMock()

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber._handle_workflow_action(parsed_action, event_data)

        # Verify fallback was called for missing instruction
        subscriber._handle_notification_fallback.assert_called_once_with(event_data)
        workflow_engine.start_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_workflow_action_error(self):
        """Test error handling in workflow action."""
        workflow_engine = Mock()
        universal_agent = Mock()

        # Mock workflow creation error
        workflow_engine.start_workflow = AsyncMock(
            side_effect=Exception("Workflow creation failed")
        )

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        parsed_action = {"workflow_instruction": "Create summary report"}

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        subscriber._handle_notification_fallback = AsyncMock()

        # Execute handling
        await subscriber._handle_workflow_action(parsed_action, event_data)

        # Verify fallback was called on error
        workflow_engine.start_workflow.assert_called_once()
        subscriber._handle_notification_fallback.assert_called_once_with(event_data)

    @pytest.mark.asyncio
    async def test_handle_direct_action(self):
        """Test handling direct action."""
        workflow_engine = Mock()
        universal_agent = Mock()
        role_mock = Mock()

        # Mock role execution
        universal_agent.assume_role.return_value = role_mock
        role_mock.execute_task = AsyncMock(return_value={"status": "success"})

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        parsed_action = {
            "target_role": "weather",
            "action_params": {"location": "Seattle"},
        }

        event_data = {
            "original_request": "Get weather report",
            "execution_context": {"user_id": "user123"},
        }

        # Execute handling
        await subscriber._handle_direct_action(parsed_action, event_data)

        # Verify role action was executed
        universal_agent.assume_role.assert_called_once_with("weather")
        role_mock.execute_task.assert_called_once_with(
            {"location": "Seattle"}, context=event_data["execution_context"]
        )

    @pytest.mark.asyncio
    async def test_handle_direct_action_error(self):
        """Test error handling in direct action."""
        workflow_engine = Mock()
        universal_agent = Mock()
        role_mock = Mock()

        # Mock role execution error
        universal_agent.assume_role.return_value = role_mock
        role_mock.execute_task = AsyncMock(
            side_effect=Exception("Role execution failed")
        )

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        parsed_action = {
            "target_role": "weather",
            "action_params": {"location": "Seattle"},
        }

        event_data = {
            "original_request": "Get weather report",
            "execution_context": {"user_id": "user123"},
        }

        subscriber._handle_notification_fallback = AsyncMock()

        # Execute handling
        await subscriber._handle_direct_action(parsed_action, event_data)

        # Verify fallback was called on error
        universal_agent.assume_role.assert_called_once()
        role_mock.execute_task.assert_called_once()
        subscriber._handle_notification_fallback.assert_called_once_with(event_data)

    @pytest.mark.asyncio
    async def test_handle_notification_fallback(self):
        """Test notification fallback handling."""
        workflow_engine = Mock()
        universal_agent = Mock()

        subscriber = EventSubscriber(
            role_name="timer",
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
        )

        event_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123"},
        }

        # Execute fallback
        await subscriber._handle_notification_fallback(event_data)

        # No assertions needed for now as the method mainly logs
        # In a real implementation, we would verify notification sending

    @pytest.mark.asyncio
    async def test_custom_timer_event_subscriber(self):
        """Test custom TimerEventSubscriber implementation."""
        workflow_engine = Mock()
        universal_agent = Mock()
        communication_manager = AsyncMock()

        from common.communication_manager import ChannelType
        from roles.shared_tools.event_subscriber import TimerEventSubscriber

        subscriber = TimerEventSubscriber(
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
            communication_manager=communication_manager,
        )

        # Mock _parse_delayed_action to return notification action
        parsed_action = {
            "action_type": ActionType.NOTIFICATION.value,
            "workflow_instruction": "Timer expired notification",
        }
        subscriber._parse_delayed_action = AsyncMock(return_value=parsed_action)

        timer_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123", "channel": "general"},
        }

        # Execute handling
        await subscriber.handle_timer_expiry_with_context(timer_data)

        # Verify notification was sent
        communication_manager.send_notification.assert_called_once_with(
            message="Timer expired notification",
            channels=["slack"],
            recipient="general",
            metadata=timer_data["execution_context"],
        )

    @pytest.mark.asyncio
    async def test_custom_timer_event_subscriber_workflow_type(self):
        """Test custom TimerEventSubscriber with workflow action type."""
        workflow_engine = Mock()
        universal_agent = Mock()
        communication_manager = Mock()

        from roles.shared_tools.event_subscriber import TimerEventSubscriber

        subscriber = TimerEventSubscriber(
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
            communication_manager=communication_manager,
        )

        # Mock _parse_delayed_action to return workflow action
        parsed_action = {
            "action_type": ActionType.WORKFLOW.value,
            "workflow_instruction": "Create workflow for timer",
        }
        subscriber._parse_delayed_action = AsyncMock(return_value=parsed_action)
        subscriber._handle_workflow_action = AsyncMock()

        timer_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123", "channel": "general"},
        }

        # Execute handling
        await subscriber.handle_timer_expiry_with_context(timer_data)

        # Verify workflow action was handled
        subscriber._handle_workflow_action.assert_called_once_with(
            parsed_action, timer_data
        )
        communication_manager.send_notification.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_timer_event_subscriber_error(self):
        """Test error handling in custom TimerEventSubscriber."""
        workflow_engine = Mock()
        universal_agent = Mock()
        communication_manager = AsyncMock()

        from common.communication_manager import ChannelType
        from roles.shared_tools.event_subscriber import TimerEventSubscriber

        subscriber = TimerEventSubscriber(
            workflow_engine=workflow_engine,
            universal_agent=universal_agent,
            communication_manager=communication_manager,
        )

        # Mock _parse_delayed_action to throw error
        subscriber._parse_delayed_action = AsyncMock(
            side_effect=Exception("Parsing error")
        )

        timer_data = {
            "original_request": "Remind me in 5 minutes",
            "execution_context": {"user_id": "user123", "channel": "general"},
        }

        # Execute handling
        await subscriber.handle_timer_expiry_with_context(timer_data)

        # Verify fallback notification was sent
        communication_manager.send_notification.assert_called_once()
        assert (
            "Timer reminder"
            in communication_manager.send_notification.call_args[1]["message"]
        )
        assert communication_manager.send_notification.call_args[1]["channels"] == [
            "slack"
        ]
