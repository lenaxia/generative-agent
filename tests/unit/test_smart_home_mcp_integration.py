"""Tests for Smart Home role with Home Assistant MCP integration.

This test suite validates the enhanced smart home role that integrates with
Home Assistant via MCP (Model Context Protocol) for real device control.
"""

import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.enhanced_event_context import LLMSafeEventContext
from common.intents import AuditIntent, NotificationIntent
from roles.core_smart_home import (
    HOME_ASSISTANT_DOMAINS,
    HOME_ASSISTANT_SERVICES,
    ROLE_CONFIG,
    HomeAssistantServiceIntent,
    HomeAssistantStateIntent,
    SmartHomeControlIntent,
    create_smart_home_error_intent,
    fetch_home_assistant_entities,
    format_home_assistant_response,
    get_available_services,
    ha_call_service,
    ha_get_state,
    ha_list_entities,
    handle_device_discovery,
    handle_smart_home_request,
    process_home_assistant_service_intent,
    process_home_assistant_state_intent,
    process_smart_home_control_intent,
    register_role,
    validate_entity_id,
)


class TestSmartHomeRoleConfiguration:
    """Test smart home role configuration and metadata."""

    def test_role_config_structure(self):
        """Test that role config follows LLM-safe patterns."""
        assert ROLE_CONFIG["name"] == "smart_home"
        assert ROLE_CONFIG["version"] == "4.0.0"
        assert ROLE_CONFIG["llm_type"] == "WEAK"
        assert ROLE_CONFIG["fast_reply"] is True

        # Test MCP integration configuration
        tools_config = ROLE_CONFIG["tools"]
        assert tools_config["automatic"] is True
        assert tools_config["include_builtin"] is False
        assert "mcp_integration" in tools_config

        mcp_config = tools_config["mcp_integration"]
        assert mcp_config["enabled"] is True
        assert "home_assistant" in mcp_config["preferred_servers"]
        assert "call_service" in mcp_config["tool_filters"]

    def test_role_parameters_schema(self):
        """Test role parameter schema for router integration."""
        params = ROLE_CONFIG["parameters"]

        # Test required action parameter
        action_param = params["action"]
        assert action_param["type"] == "string"
        assert action_param["required"] is True
        assert "turn_on" in action_param["examples"]

        # Test optional entity_id parameter
        entity_param = params["entity_id"]
        assert entity_param["required"] is False
        assert "light.living_room" in entity_param["examples"]

    def test_system_prompt_includes_mcp_tools(self):
        """Test that system prompt includes Home Assistant MCP tools."""
        system_prompt = ROLE_CONFIG["prompts"]["system"]
        assert "ha_call_service" in system_prompt
        assert "ha_get_state" in system_prompt
        assert "ha_list_entities" in system_prompt
        assert "Home Assistant" in system_prompt


class TestHomeAssistantIntents:
    """Test Home Assistant specific intent definitions."""

    def test_home_assistant_service_intent_validation(self):
        """Test HomeAssistantServiceIntent validation."""
        # Valid intent
        valid_intent = HomeAssistantServiceIntent(
            domain="light",
            service="turn_on",
            entity_id="light.living_room",
            service_data={"brightness": 255},
        )
        assert valid_intent.validate() is True

        # Invalid intent - missing domain
        invalid_intent = HomeAssistantServiceIntent(domain="", service="turn_on")
        assert invalid_intent.validate() is False

        # Invalid intent - missing service
        invalid_intent2 = HomeAssistantServiceIntent(domain="light", service="")
        assert invalid_intent2.validate() is False

    def test_home_assistant_state_intent_validation(self):
        """Test HomeAssistantStateIntent validation."""
        # Valid get_state intent
        valid_intent = HomeAssistantStateIntent(
            entity_id="light.living_room", operation="get_state"
        )
        assert valid_intent.validate() is True

        # Valid list_entities intent
        valid_intent2 = HomeAssistantStateIntent(
            domain="light", operation="list_entities"
        )
        assert valid_intent2.validate() is True

        # Invalid operation
        invalid_intent = HomeAssistantStateIntent(operation="invalid_operation")
        assert invalid_intent.validate() is False

    def test_smart_home_control_intent_validation(self):
        """Test SmartHomeControlIntent validation."""
        # Valid intent
        valid_intent = SmartHomeControlIntent(
            action="control_device",
            target_entity="light.living_room",
            parameters={"brightness": 255},
        )
        assert valid_intent.validate() is True

        # Invalid action
        invalid_intent = SmartHomeControlIntent(action="invalid_action")
        assert invalid_intent.validate() is False


class TestEventHandlers:
    """Test pure function event handlers."""

    @pytest.fixture
    def mock_context(self):
        """Create mock LLMSafeEventContext."""
        context = MagicMock(spec=LLMSafeEventContext)
        context.user_id = "U123456"
        context.channel_id = "slack:C123456"
        context.get_safe_channel.return_value = "slack:C123456"
        context.to_dict.return_value = {
            "user_id": "U123456",
            "channel_id": "slack:C123456",
            "timestamp": time.time(),
        }
        return context

    def test_handle_smart_home_request_success(self, mock_context):
        """Test successful smart home request handling."""
        event_data = {
            "action": "turn_on",
            "entity_id": "light.living_room",
            "parameters": {"brightness": 255},
        }

        intents = handle_smart_home_request(event_data, mock_context)

        assert len(intents) == 2
        assert isinstance(intents[0], SmartHomeControlIntent)
        assert isinstance(intents[1], AuditIntent)

        control_intent = intents[0]
        assert control_intent.action == "control_device"
        assert control_intent.target_entity == "light.living_room"
        assert control_intent.parameters == {"brightness": 255}

    def test_handle_smart_home_request_error(self, mock_context):
        """Test smart home request handling with error."""
        # Force an exception by patching the parsing function
        with patch(
            "roles.core_smart_home._parse_smart_home_event_data",
            side_effect=Exception("Test error"),
        ):
            intents = handle_smart_home_request({}, mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], NotificationIntent)
        assert intents[0].notification_type == "error"

    def test_handle_device_discovery_success(self, mock_context):
        """Test successful device discovery handling."""
        event_data = {"domain": "light"}

        intents = handle_device_discovery(event_data, mock_context)

        assert len(intents) == 2
        assert isinstance(intents[0], HomeAssistantStateIntent)
        assert isinstance(intents[1], AuditIntent)

        state_intent = intents[0]
        assert state_intent.domain == "light"
        assert state_intent.operation == "list_entities"

    def test_handle_device_discovery_error(self, mock_context):
        """Test device discovery handling with error."""
        # Force an exception by passing invalid data
        with patch(
            "roles.core_smart_home._parse_device_discovery_event",
            side_effect=Exception("Test error"),
        ):
            intents = handle_device_discovery({}, mock_context)

        assert len(intents) == 1
        assert isinstance(intents[0], NotificationIntent)
        assert intents[0].notification_type == "warning"


class TestMCPIntegratedTools:
    """Test MCP-integrated Home Assistant tools."""

    def test_ha_call_service_success(self):
        """Test successful Home Assistant service call tool."""
        result = ha_call_service(
            domain="light",
            service="turn_on",
            entity_id="light.living_room",
            brightness=255,
        )

        assert result["success"] is True
        assert "intent" in result

        intent_data = result["intent"]
        assert intent_data["type"] == "HomeAssistantServiceIntent"
        assert intent_data["domain"] == "light"
        assert intent_data["service"] == "turn_on"
        assert intent_data["entity_id"] == "light.living_room"
        assert intent_data["service_data"]["brightness"] == 255

    def test_ha_call_service_validation_error(self):
        """Test Home Assistant service call with validation error."""
        result = ha_call_service(domain="", service="")

        assert result["success"] is False
        assert "error" in result
        assert "Domain and service are required" in result["error"]

    def test_ha_get_state_success(self):
        """Test successful Home Assistant state query tool."""
        result = ha_get_state(entity_id="light.living_room")

        assert result["success"] is True
        assert "intent" in result

        intent_data = result["intent"]
        assert intent_data["type"] == "HomeAssistantStateIntent"
        assert intent_data["entity_id"] == "light.living_room"
        assert intent_data["operation"] == "get_state"

    def test_ha_get_state_validation_error(self):
        """Test Home Assistant state query with validation error."""
        result = ha_get_state(entity_id="")

        assert result["success"] is False
        assert "error" in result
        assert "Entity ID is required" in result["error"]

    def test_ha_list_entities_success(self):
        """Test successful Home Assistant entity listing tool."""
        result = ha_list_entities(domain="light")

        assert result["success"] is True
        assert "intent" in result

        intent_data = result["intent"]
        assert intent_data["type"] == "HomeAssistantStateIntent"
        assert intent_data["domain"] == "light"
        assert intent_data["operation"] == "list_entities"

    def test_ha_list_entities_no_domain(self):
        """Test Home Assistant entity listing without domain filter."""
        result = ha_list_entities()

        assert result["success"] is True
        intent_data = result["intent"]
        assert intent_data["domain"] is None


class TestIntentProcessors:
    """Test intent processor functions."""

    @pytest.mark.asyncio
    async def test_process_home_assistant_service_intent(self):
        """Test Home Assistant service intent processing."""
        intent = HomeAssistantServiceIntent(
            domain="light",
            service="turn_on",
            entity_id="light.living_room",
            service_data={"brightness": 255},
            user_id="U123456",
            event_context={"test": "context"},
        )

        # Should not raise exception
        await process_home_assistant_service_intent(intent)

    @pytest.mark.asyncio
    async def test_process_home_assistant_state_intent_get_state(self):
        """Test Home Assistant state intent processing for get_state."""
        intent = HomeAssistantStateIntent(
            entity_id="light.living_room", operation="get_state", user_id="U123456"
        )

        # Should not raise exception
        await process_home_assistant_state_intent(intent)

    @pytest.mark.asyncio
    async def test_process_home_assistant_state_intent_list_entities(self):
        """Test Home Assistant state intent processing for list_entities."""
        intent = HomeAssistantStateIntent(
            domain="light", operation="list_entities", user_id="U123456"
        )

        # Should not raise exception
        await process_home_assistant_state_intent(intent)

    @pytest.mark.asyncio
    async def test_process_smart_home_control_intent(self):
        """Test smart home control intent processing."""
        intent = SmartHomeControlIntent(
            action="control_device",
            target_entity="light.living_room",
            parameters={"brightness": 255},
            user_id="U123456",
        )

        # Should not raise exception
        await process_smart_home_control_intent(intent)


class TestPrePostProcessors:
    """Test pre and post processors for MCP integration."""

    @pytest.mark.asyncio
    async def test_fetch_home_assistant_entities_success(self):
        """Test successful Home Assistant entity fetching."""
        parameters = {"domain": "light", "entity_id": "light.living_room"}

        result = await fetch_home_assistant_entities(parameters)

        assert result["success"] is True
        assert "entities" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_fetch_home_assistant_entities_error(self):
        """Test Home Assistant entity fetching with error."""
        # Force an error by patching logger to raise exception
        with patch(
            "roles.core_smart_home.logger.info", side_effect=Exception("Test error")
        ):
            result = await fetch_home_assistant_entities({})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_format_home_assistant_response(self):
        """Test Home Assistant response formatting."""
        llm_result = "The lights have been turned on."
        context = {}
        pre_data = {"entities": []}

        result = await format_home_assistant_response(llm_result, context, pre_data)

        # Should return the original result (formatting not fully implemented)
        assert result == llm_result


class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_get_available_services_known_domain(self):
        """Test getting available services for known domain."""
        services = get_available_services("light")

        assert "turn_on" in services
        assert "turn_off" in services
        assert "toggle" in services
        assert "brightness_increase" in services

    def test_get_available_services_unknown_domain(self):
        """Test getting available services for unknown domain."""
        services = get_available_services("unknown_domain")

        # Should return default services
        assert "turn_on" in services
        assert "turn_off" in services
        assert "toggle" in services

    def test_validate_entity_id_valid(self):
        """Test entity ID validation with valid IDs."""
        assert validate_entity_id("light.living_room") is True
        assert validate_entity_id("switch.kitchen") is True
        assert validate_entity_id("climate.thermostat") is True

    def test_validate_entity_id_invalid(self):
        """Test entity ID validation with invalid IDs."""
        assert validate_entity_id("") is False
        assert validate_entity_id("invalid") is False
        assert validate_entity_id("unknown_domain.device") is False

    def test_create_smart_home_error_intent(self):
        """Test smart home error intent creation."""
        context = MagicMock(spec=LLMSafeEventContext)
        context.user_id = "U123456"
        context.get_safe_channel.return_value = "slack:C123456"
        context.to_dict.return_value = {"test": "context"}

        error = Exception("Test error")
        intents = create_smart_home_error_intent(error, context)

        assert len(intents) == 2
        assert isinstance(intents[0], NotificationIntent)
        assert isinstance(intents[1], AuditIntent)

        notification = intents[0]
        assert notification.notification_type == "error"
        assert "Test error" in notification.message


class TestRoleRegistration:
    """Test role registration and auto-discovery."""

    def test_register_role_structure(self):
        """Test that role registration returns correct structure."""
        registration = register_role()

        assert "config" in registration
        assert "event_handlers" in registration
        assert "tools" in registration
        assert "intents" in registration
        assert "pre_processors" in registration
        assert "post_processors" in registration

        # Test event handlers
        handlers = registration["event_handlers"]
        assert "SMART_HOME_REQUEST" in handlers
        assert "DEVICE_DISCOVERY" in handlers

        # Test tools
        tools = registration["tools"]
        assert ha_call_service in tools
        assert ha_get_state in tools
        assert ha_list_entities in tools

        # Test intents
        intents = registration["intents"]
        assert HomeAssistantServiceIntent in intents
        assert HomeAssistantStateIntent in intents
        assert SmartHomeControlIntent in intents

    def test_home_assistant_constants(self):
        """Test Home Assistant domain and service constants."""
        # Test domains
        assert "light" in HOME_ASSISTANT_DOMAINS
        assert "switch" in HOME_ASSISTANT_DOMAINS
        assert "climate" in HOME_ASSISTANT_DOMAINS

        # Test services
        assert "light" in HOME_ASSISTANT_SERVICES
        assert "turn_on" in HOME_ASSISTANT_SERVICES["light"]
        assert "set_temperature" in HOME_ASSISTANT_SERVICES["climate"]


class TestMCPIntegrationPatterns:
    """Test MCP integration patterns and compliance."""

    def test_tools_return_intents_not_direct_calls(self):
        """Test that tools return intents instead of making direct MCP calls."""
        # All tools should return intent data, not make direct API calls
        result = ha_call_service("light", "turn_on", "light.living_room")
        assert "intent" in result
        assert result["intent"]["type"] == "HomeAssistantServiceIntent"

        result = ha_get_state("light.living_room")
        assert "intent" in result
        assert result["intent"]["type"] == "HomeAssistantStateIntent"

        result = ha_list_entities("light")
        assert "intent" in result
        assert result["intent"]["type"] == "HomeAssistantStateIntent"

    def test_intent_processors_handle_mcp_integration(self):
        """Test that intent processors are designed for MCP integration."""
        # Intent processors should be async and handle MCP client calls
        # This is tested through the async nature of the functions
        import inspect

        assert inspect.iscoroutinefunction(process_home_assistant_service_intent)
        assert inspect.iscoroutinefunction(process_home_assistant_state_intent)
        assert inspect.iscoroutinefunction(process_smart_home_control_intent)

    def test_role_config_mcp_compliance(self):
        """Test that role configuration is MCP compliant."""
        tools_config = ROLE_CONFIG["tools"]

        # Should have MCP integration enabled
        assert tools_config["mcp_integration"]["enabled"] is True

        # Should prefer home_assistant server
        assert "home_assistant" in tools_config["mcp_integration"]["preferred_servers"]

        # Should have appropriate tool filters
        filters = tools_config["mcp_integration"]["tool_filters"]
        assert "call_service" in filters
        assert "get_state" in filters
        assert "list_entities" in filters
