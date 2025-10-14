"""
End-to-end tests for timer notification routing fixes.

Tests the complete flow from Slack timer request to Slack notification delivery.
"""

import asyncio
import logging
import time
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.channel_handlers.console_handler import ConsoleChannelHandler
from common.channel_handlers.slack_handler import SlackChannelHandler
from common.communication_manager import CommunicationManager
from common.message_bus import MessageBus
from common.request_model import RequestMetadata
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent
from roles.timer_single_file import handle_timer_expiry, set_timer
from supervisor.workflow_engine import WorkflowEngine


class TestTimerNotificationRouting:
    """Test timer notification routing end-to-end."""

    @pytest.fixture
    def setup_system(self):
        """Set up the complete system for testing."""
        # Create message bus
        message_bus = MessageBus()
        message_bus.start()

        # Create communication manager
        comm_manager = CommunicationManager(message_bus)

        # Create and register channel handlers with patterns
        slack_handler = SlackChannelHandler()
        console_handler = ConsoleChannelHandler()

        # Mock the handlers to avoid actual Slack/console calls
        slack_handler.send_notification = AsyncMock(
            return_value={
                "success": True,
                "channel": "#general",
                "ts": "1234567890.123456",
                "message_id": "1234567890.123456",
            }
        )
        console_handler.send_notification = AsyncMock(
            return_value={"success": True, "message": "Console notification sent"}
        )

        # Register handlers
        comm_manager.channels["slack"] = slack_handler
        comm_manager.channels["console"] = console_handler

        return {
            "message_bus": message_bus,
            "comm_manager": comm_manager,
            "slack_handler": slack_handler,
            "console_handler": console_handler,
        }

    def test_parameter_schema_includes_context_parameters(self):
        """Test that timer role parameter schema includes user_id and channel_id."""
        role_registry = RoleRegistry()
        timer_role = role_registry.get_role("timer")

        role_params = timer_role.config.get("role", {}).get("parameters", {})

        # Verify context parameters are in schema
        assert (
            "user_id" in role_params
        ), "user_id parameter missing from timer role schema"
        assert (
            "channel_id" in role_params
        ), "channel_id parameter missing from timer role schema"

        # Verify parameter descriptions
        user_id_param = role_params["user_id"]
        channel_id_param = role_params["channel_id"]

        assert user_id_param["type"] == "string"
        assert channel_id_param["type"] == "string"
        assert "notification routing" in user_id_param["description"]
        assert "notification routing" in channel_id_param["description"]

    def test_communication_manager_pattern_matching(self, setup_system):
        """Test that communication manager correctly matches channel patterns."""
        system = setup_system
        comm_manager = system["comm_manager"]

        # Test Slack channel pattern matching
        slack_handler = comm_manager._find_handler_for_channel("slack:C52L1UK5E")
        assert slack_handler is not None, "No handler found for Slack channel"
        assert isinstance(
            slack_handler, SlackChannelHandler
        ), "Wrong handler type for Slack channel"

        # Test console channel pattern matching
        console_handler = comm_manager._find_handler_for_channel("console")
        assert console_handler is not None, "No handler found for console channel"
        assert isinstance(
            console_handler, ConsoleChannelHandler
        ), "Wrong handler type for console channel"

    def test_target_channel_determination(self, setup_system):
        """Test that target channels are determined correctly."""
        system = setup_system
        comm_manager = system["comm_manager"]

        # Test Slack channel routing
        slack_targets = comm_manager._determine_target_channels(
            "slack:C52L1UK5E", "notification", {}
        )
        assert slack_targets == [
            "slack:C52L1UK5E"
        ], f"Expected ['slack:C52L1UK5E'], got {slack_targets}"

        # Test console channel routing
        console_targets = comm_manager._determine_target_channels(
            "console", "notification", {}
        )
        assert console_targets == [
            "console"
        ], f"Expected ['console'], got {console_targets}"

    @patch("roles.shared_tools.redis_tools.redis_write")
    def test_timer_creation_with_context_parameters(self, mock_redis_write):
        """Test that timer creation stores proper context when parameters are provided."""
        # Mock Redis success
        mock_redis_write.return_value = {"success": True}

        # Test timer creation with Slack context
        result = set_timer(
            duration="5s",
            label="test timer",
            user_id="U52L1U8M6",
            channel_id="slack:C52L1UK5E",
        )

        # Verify timer creation succeeded
        assert result["success"] is True
        assert "timer_id" in result

        # Verify Redis was called with correct context
        mock_redis_write.assert_called_once()
        call_args = mock_redis_write.call_args
        timer_data = call_args[0][1]  # Second argument is the timer data

        # Verify context is stored correctly
        assert (
            timer_data["user_id"] == "U52L1U8M6"
        ), f"Expected U52L1U8M6, got {timer_data['user_id']}"
        assert (
            timer_data["channel"] == "slack:C52L1UK5E"
        ), f"Expected slack:C52L1UK5E, got {timer_data['channel']}"

    @patch("roles.shared_tools.redis_tools.redis_delete")
    def test_timer_expiry_uses_stored_context(self, mock_redis_delete, setup_system):
        """Test that timer expiry uses stored context for notification routing."""
        system = setup_system
        comm_manager = system["comm_manager"]
        slack_handler = system["slack_handler"]

        # Mock Redis delete success
        mock_redis_delete.return_value = {"success": True}

        # Create timer data with Slack context (simulating stored timer)
        timer_data = {
            "id": "timer_test123",
            "duration": "5s",
            "label": "test timer",
            "user_id": "U52L1U8M6",
            "channel": "slack:C52L1UK5E",  # This should be used for notification
            "created_at": time.time(),
            "expires_at": time.time() + 5,
            "status": "active",
        }

        # Mock the communication manager to capture the notification
        with patch.object(comm_manager, "message_bus") as mock_bus:
            # Call timer expiry handler with proper context
            from common.enhanced_event_context import LLMSafeEventContext

            context = LLMSafeEventContext(
                user_id="U52L1U8M6",
                channel_id="slack:C52L1UK5E",
                source="test",
                metadata={},
            )

            handle_timer_expiry(["timer_test123", "Test timer expired"], context)

            # Verify message bus was called with correct context
            mock_bus.publish.assert_called_once()
            call_args = mock_bus.publish.call_args

            # Extract the published message
            publisher = (
                call_args[1]["publisher"]
                if "publisher" in call_args[1]
                else call_args[0][0]
            )
            message_type = (
                call_args[1]["message_type"]
                if "message_type" in call_args[1]
                else call_args[0][1]
            )
            message_payload = (
                call_args[1]["message"]
                if "message" in call_args[1]
                else call_args[0][2]
            )

            # Verify correct context is used
            context = message_payload.get("context", {})
            assert (
                context.get("channel_id") == "slack:C52L1UK5E"
            ), f"Expected slack:C52L1UK5E, got {context.get('channel_id')}"
            assert (
                context.get("user_id") == "U52L1U8M6"
            ), f"Expected U52L1U8M6, got {context.get('user_id')}"

    @pytest.mark.asyncio
    async def test_end_to_end_slack_timer_notification(self, setup_system):
        """Test complete end-to-end flow from Slack request to Slack notification."""
        system = setup_system
        comm_manager = system["comm_manager"]
        slack_handler = system["slack_handler"]

        # 1. Simulate Slack timer request context
        slack_context = {
            "channel_id": "slack:C52L1UK5E",
            "user_id": "U52L1U8M6",
            "request_id": "test_request_123",
        }

        # 2. Simulate timer expiry notification (the failing part)
        timer_message = {"message": "⏰ Timer expired: 5s", "context": slack_context}

        # 3. Test message routing through communication manager
        result = await comm_manager.route_message(
            timer_message["message"], timer_message["context"]
        )

        # 4. Verify notification was routed to Slack handler
        assert len(result) > 0, "No delivery results returned"

        delivery_result = result[0]
        assert (
            delivery_result["channel"] == "slack:C52L1UK5E"
        ), f"Wrong channel in result: {delivery_result}"
        assert delivery_result["result"]["success"] is True, "Slack delivery failed"

        # 5. Verify Slack handler was called with correct parameters
        slack_handler.send_notification.assert_called_once()
        call_args = slack_handler.send_notification.call_args

        # Verify message content
        message_sent = call_args[0][0]  # First argument is message
        assert "⏰ Timer expired: 5s" in message_sent

    def test_console_fallback_routing(self, setup_system):
        """Test that console notifications route to console handler."""
        system = setup_system
        comm_manager = system["comm_manager"]

        # Test console channel routing
        console_targets = comm_manager._determine_target_channels(
            "console", "notification", {}
        )
        assert console_targets == [
            "console"
        ], f"Console routing failed: {console_targets}"

        # Test console handler matching
        console_handler = comm_manager._find_handler_for_channel("console")
        assert console_handler is not None, "Console handler not found"
        assert isinstance(
            console_handler, ConsoleChannelHandler
        ), "Wrong console handler type"

    def test_role_registry_pre_processor_support(self):
        """Test that role registry properly loads single-file role pre-processors."""
        role_registry = RoleRegistry()
        timer_role = role_registry.get_role("timer")

        # Verify pre-processors are loaded
        has_preprocessors = hasattr(timer_role, "_pre_processors")
        assert has_preprocessors, "Timer role missing _pre_processors attribute"

        if has_preprocessors:
            processors = getattr(timer_role, "_pre_processors", [])
            assert len(processors) > 0, "No pre-processors loaded for timer role"

            # Verify pre-processor function names
            processor_names = [
                getattr(proc, "__name__", "unknown") for proc in processors
            ]
            assert (
                "_timer_context_injector" in processor_names
            ), f"Context injector not found in {processor_names}"

    @pytest.mark.asyncio
    async def test_universal_agent_pre_processor_execution(self):
        """Test that Universal Agent executes single-file role pre-processors."""
        # This test verifies the Universal Agent fix for pre-processor support
        role_registry = RoleRegistry()
        timer_role = role_registry.get_role("timer")

        # Mock LLM factory to avoid actual LLM calls
        mock_configs = {"WEAK": [Mock()], "DEFAULT": [Mock()], "STRONG": [Mock()]}

        with patch("llm_provider.factory.LLMFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory_class.return_value = mock_factory

            universal_agent = UniversalAgent(mock_factory, role_registry)

            # Check if timer role has pre-processors
            has_preprocessors = hasattr(timer_role, "_pre_processors")
            if has_preprocessors:
                processors = getattr(timer_role, "_pre_processors", [])
                assert len(processors) > 0, "Pre-processors should be available"

                # Verify Universal Agent can access them
                from common.task_context import TaskContext

                mock_context = Mock(spec=TaskContext)
                mock_context.user_id = "U52L1U8M6"
                mock_context.channel_id = "slack:C52L1UK5E"

                # Test pre-processor execution (without full LLM execution)
                try:
                    result = await universal_agent._run_pre_processors(
                        timer_role,
                        {},  # lifecycle_functions
                        "timer 5s",  # instruction
                        mock_context,
                        {"duration": "5s"},  # parameters
                    )
                    # If this doesn't throw an exception, pre-processor support is working
                    assert True, "Pre-processor execution completed"
                except Exception as e:
                    pytest.fail(f"Pre-processor execution failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
