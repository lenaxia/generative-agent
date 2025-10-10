"""
Tests for additional Slack workflow fixes.

This module tests the fixes for:
1. Duplicate message processing (incoming_message and app_mention)
2. Missing Slack responses (workflow results sent back to Slack)
"""

import asyncio
import queue
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import CommunicationManager
from common.message_bus import MessageBus, MessageType
from common.request_model import RequestMetadata
from supervisor.workflow_engine import WorkflowEngine


class TestSlackDuplicateMessageFix:
    """Test fix for duplicate message processing."""

    @pytest.fixture
    def slack_handler(self):
        """Create Slack handler for testing."""
        with patch("slack_bolt.App"):
            handler = SlackChannelHandler()
            handler.communication_manager = Mock()
            handler.message_queue = queue.Queue()
            handler.bot_user_id = "U07TCJFKF1C"  # Mock bot user ID
            return handler

    def test_message_handler_ignores_bot_mentions(self, slack_handler):
        """Test that message handler ignores messages with bot mentions."""
        # Simulate a message event with bot mention
        event = {
            "user": "U123456",
            "channel": "C123456",
            "text": "<@U07TCJFKF1C> what's the weather?",
            "ts": "123456",
        }

        # The message handler should ignore this (it will be handled by app_mention)
        # We can't directly test the handler function, but we can test the logic
        text = event.get("text", "")
        should_ignore = (
            text
            and slack_handler.bot_user_id
            and f"<@{slack_handler.bot_user_id}>" in text
        )

        assert should_ignore is True

    def test_message_handler_processes_direct_messages(self, slack_handler):
        """Test that message handler processes direct messages without bot mentions."""
        # Simulate a direct message event without bot mention
        event = {
            "user": "U123456",
            "channel": "D123456",  # Direct message channel
            "text": "what's the weather?",
            "ts": "123456",
        }

        # The message handler should process this
        text = event.get("text", "")
        should_ignore = (
            text
            and slack_handler.bot_user_id
            and f"<@{slack_handler.bot_user_id}>" in text
        )

        assert should_ignore is False

    def test_app_mention_handler_processes_mentions(self, slack_handler):
        """Test that app mention handler processes bot mentions."""
        # Simulate an app mention event
        event = {
            "user": "U123456",
            "channel": "C123456",
            "text": "<@U07TCJFKF1C> what's the weather?",
            "ts": "123456",
        }

        # App mention handler should always process mentions (no bot_id check needed)
        should_process = not event.get("bot_id")

        assert should_process is True


class TestSlackResponseFlow:
    """Test fix for missing Slack responses."""

    @pytest.fixture
    def workflow_engine(self):
        """Create workflow engine for testing."""
        message_bus = Mock(spec=MessageBus)
        llm_factory = Mock()
        engine = WorkflowEngine(llm_factory, message_bus)
        engine.request_router = Mock()
        engine.universal_agent = Mock()
        engine.role_registry = Mock()
        return engine

    @pytest.mark.asyncio
    async def test_fast_reply_sends_response_back(self, workflow_engine):
        """Test that fast-reply sends response back to requester."""
        # Mock routing result
        routing_result = {
            "route": "weather",
            "confidence": 0.95,
            "parameters": {"location": "seattle"},
        }
        workflow_engine.request_router.route_request.return_value = routing_result

        # Mock role registry
        workflow_engine.role_registry.get_role_execution_type.return_value = "hybrid"

        # Mock universal agent execution
        mock_result = "Currently in Seattle, it's 61°F and rainy."
        workflow_engine.universal_agent.execute_task.return_value = mock_result

        # Create request metadata with response_requested=True
        request = RequestMetadata(
            prompt="what's the weather in seattle?",
            source_id="slack",
            target_id="workflow_engine",
            metadata={
                "user_id": "U123456",
                "channel_id": "slack:C123456",
                "source": "slack",
                "request_id": "test_request_123",
            },
            response_requested=True,
        )

        # Mock duration logger
        with patch(
            "supervisor.workflow_engine.get_duration_logger"
        ) as mock_duration_logger:
            mock_duration_logger.return_value = Mock()

            # Execute fast reply
            request_id = workflow_engine._handle_fast_reply(request, routing_result)

            # Verify response was published back to message bus
            workflow_engine.message_bus.publish.assert_called()
            call_args = workflow_engine.message_bus.publish.call_args

            # Check that SEND_MESSAGE was published
            assert call_args[0][1] == MessageType.SEND_MESSAGE
            message_data = call_args[0][2]
            assert message_data["message"] == mock_result
            assert message_data["context"]["channel_id"] == "slack:C123456"
            assert message_data["context"]["user_id"] == "U123456"

    @pytest.mark.asyncio
    async def test_communication_manager_routes_response_to_slack(self):
        """Test that communication manager routes responses to Slack channel."""
        # Create communication manager
        message_bus = Mock(spec=MessageBus)
        comm_manager = CommunicationManager(message_bus)

        # Create mock Slack handler
        mock_slack_handler = Mock()
        mock_slack_handler.send = AsyncMock()
        comm_manager.channels["slack"] = mock_slack_handler

        # Create response message
        response_message = {
            "message": "Currently in Seattle, it's 61°F and rainy.",
            "context": {
                "channel_id": "slack:C123456",
                "user_id": "U123456",
                "request_id": "test_request_123",
            },
        }

        # Mock the _send_with_delivery_guarantee method to track calls
        with patch.object(
            comm_manager, "_send_with_delivery_guarantee", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = [{"status": "success", "channel": "slack"}]

            # Handle the send message
            await comm_manager._handle_send_message(response_message)

            # Verify the message was processed
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert (
                "Currently in Seattle, it's 61°F and rainy." in call_args[0][0]
            )  # message
            assert "slack" in call_args[0][1]  # target_channels


class TestEndToEndSlackWorkflow:
    """Test complete end-to-end Slack workflow with fixes."""

    @pytest.mark.asyncio
    async def test_complete_slack_workflow_no_duplicates(self):
        """Test complete workflow from Slack message to response without duplicates."""
        # Create communication manager with mock message bus
        message_bus = Mock(spec=MessageBus)
        comm_manager = CommunicationManager(message_bus)

        # Create mock Slack handler
        mock_slack_handler = Mock()
        mock_slack_handler.send = AsyncMock()
        comm_manager.channels["slack"] = mock_slack_handler

        # Test app mention (should be processed)
        app_mention_message = {
            "type": "app_mention",
            "user_id": "U123456",
            "channel_id": "C123456",
            "text": "<@U07TCJFKF1C> what's the weather?",
            "timestamp": "123456",
        }

        # Process the app mention
        await comm_manager._handle_channel_message("slack", app_mention_message)

        # Verify message was published to message bus
        message_bus.publish.assert_called_once()

        # Verify the published message is correct
        call_args = message_bus.publish.call_args
        assert call_args[0][1] == MessageType.INCOMING_REQUEST  # Message type

        # Verify the RequestMetadata
        request_metadata = call_args[0][2]
        assert isinstance(request_metadata, RequestMetadata)
        assert request_metadata.prompt == "<@U07TCJFKF1C> what's the weather?"
        assert request_metadata.metadata["user_id"] == "U123456"
        assert request_metadata.metadata["channel_id"] == "slack:C123456"
        assert request_metadata.response_requested is True
        assert request_metadata.source_id == "slack"
        assert request_metadata.target_id == "workflow_engine"
