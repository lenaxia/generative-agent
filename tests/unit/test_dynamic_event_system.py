"""
Unit tests for the Dynamic Event-Driven Role Architecture components.

Tests the core infrastructure including EventHandlerLLM, MessageTypeRegistry,
and enhanced MessageBus functionality.
"""

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.event_handler_llm import EventHandlerLLM
from common.message_bus import EventSchema, MessageBus, MessageTypeRegistry
from llm_provider.factory import LLMFactory, LLMType


class TestEventSchema:
    """Test EventSchema validation functionality."""

    def test_event_schema_creation(self):
        """Test creating an EventSchema with proper attributes."""
        schema = {
            "timer_id": {"type": "string", "required": True},
            "original_request": {"type": "string", "required": True},
        }
        event_schema = EventSchema("TIMER_EXPIRED", schema, "Timer expiry event")

        assert event_schema.event_type == "TIMER_EXPIRED"
        assert event_schema.schema == schema
        assert event_schema.description == "Timer expiry event"

    def test_event_schema_validation_success(self):
        """Test successful validation of event data."""
        schema = {
            "timer_id": {"type": "string", "required": True},
            "optional_field": {"type": "string", "required": False},
        }
        event_schema = EventSchema("TEST_EVENT", schema)

        # Valid data with required field
        valid_data = {"timer_id": "test123", "optional_field": "value"}
        assert event_schema.validate(valid_data) is True

        # Valid data without optional field
        valid_data_minimal = {"timer_id": "test123"}
        assert event_schema.validate(valid_data_minimal) is True

    def test_event_schema_validation_failure(self):
        """Test validation failure when required fields are missing."""
        schema = {
            "timer_id": {"type": "string", "required": True},
            "required_field": {"type": "string", "required": True},
        }
        event_schema = EventSchema("TEST_EVENT", schema)

        # Missing required field
        invalid_data = {"timer_id": "test123"}
        assert event_schema.validate(invalid_data) is False

        # Empty data
        assert event_schema.validate({}) is False


class TestMessageTypeRegistry:
    """Test MessageTypeRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initializes with core system events."""
        registry = MessageTypeRegistry()

        # Should have core events registered
        assert "WORKFLOW_STARTED" in registry._registered_types
        assert "WORKFLOW_COMPLETED" in registry._registered_types
        assert "TASK_STARTED" in registry._registered_types

        # Should have system as publisher for core events
        assert "system" in registry._publishers.get("WORKFLOW_STARTED", [])

    def test_register_event_type(self):
        """Test registering new event types."""
        registry = MessageTypeRegistry()

        schema = {"timer_id": {"type": "string", "required": True}}
        registry.register_event_type("TIMER_EXPIRED", "timer", schema, "Timer expiry")

        assert "TIMER_EXPIRED" in registry._registered_types
        assert "timer" in registry._publishers["TIMER_EXPIRED"]
        assert "TIMER_EXPIRED" in registry._schemas
        assert registry._schemas["TIMER_EXPIRED"].description == "Timer expiry"

    def test_register_subscription(self):
        """Test registering event subscriptions."""
        registry = MessageTypeRegistry()

        registry.register_subscription("TIMER_EXPIRED", "smart_home")

        assert "smart_home" in registry._subscribers["TIMER_EXPIRED"]

    def test_validate_event_data_with_schema(self):
        """Test event data validation when schema exists."""
        registry = MessageTypeRegistry()

        schema = {"timer_id": {"type": "string", "required": True}}
        registry.register_event_type("TIMER_EXPIRED", "timer", schema)

        # Valid data
        valid_data = {"timer_id": "test123", "extra_field": "allowed"}
        assert registry.validate_event_data("TIMER_EXPIRED", valid_data) is True

        # Invalid data (missing required field)
        invalid_data = {"extra_field": "not_enough"}
        assert registry.validate_event_data("TIMER_EXPIRED", invalid_data) is False

    def test_validate_event_data_without_schema(self):
        """Test event data validation when no schema exists."""
        registry = MessageTypeRegistry()

        # Should return True for any data when no schema exists
        assert registry.validate_event_data("UNKNOWN_EVENT", {"any": "data"}) is True

    def test_get_event_documentation(self):
        """Test getting complete event system documentation."""
        registry = MessageTypeRegistry()

        # Register a custom event
        schema = {"timer_id": {"type": "string", "required": True}}
        registry.register_event_type("TIMER_EXPIRED", "timer", schema)
        registry.register_subscription("TIMER_EXPIRED", "smart_home")

        docs = registry.get_event_documentation()

        assert "registered_events" in docs
        assert "publishers" in docs
        assert "subscribers" in docs
        assert "schemas" in docs

        assert "TIMER_EXPIRED" in docs["registered_events"]
        assert "timer" in docs["publishers"]["TIMER_EXPIRED"]
        assert "smart_home" in docs["subscribers"]["TIMER_EXPIRED"]


class TestEventHandlerLLM:
    """Test EventHandlerLLM utility functionality."""

    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        mock_model = AsyncMock()
        mock_model.invoke = AsyncMock(return_value="Test response")
        factory.create_strands_model = Mock(return_value=mock_model)
        return factory

    @pytest.fixture
    def event_context(self):
        """Create sample event context data."""
        return {
            "timer_id": "test123",
            "original_request": "turn on the lights",
            "execution_context": {
                "user_id": "U123456",
                "channel": "#general",
                "device_context": {"room": "bedroom", "device_id": "echo_dot"},
                "timestamp": "2025-01-01T22:30:00Z",
                "source": "slack",
            },
        }

    def test_event_handler_llm_initialization(self, mock_llm_factory, event_context):
        """Test EventHandlerLLM initialization."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        assert llm.llm_factory == mock_llm_factory
        assert llm.event_context == event_context
        assert llm._execution_context == event_context["execution_context"]

    @pytest.mark.asyncio
    async def test_invoke_basic(self, mock_llm_factory, event_context):
        """Test basic LLM invocation."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        response = await llm.invoke("Test prompt")

        assert response == "Test response"
        mock_llm_factory.create_strands_model.assert_called_once_with(LLMType.WEAK)

    @pytest.mark.asyncio
    async def test_invoke_with_model_type(self, mock_llm_factory, event_context):
        """Test LLM invocation with specific model type."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        await llm.invoke("Test prompt", model_type="STRONG")

        mock_llm_factory.create_strands_model.assert_called_with(LLMType.STRONG)

    @pytest.mark.asyncio
    async def test_invoke_with_context_merging(self, mock_llm_factory, event_context):
        """Test LLM invocation with context merging."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)
        mock_model = mock_llm_factory.create_strands_model.return_value

        additional_context = {"extra_key": "extra_value"}
        await llm.invoke("Test prompt", context=additional_context)

        # Verify the prompt includes merged context
        call_args = mock_model.invoke.call_args[0][0]
        assert "user_id" in call_args  # From event context
        assert "extra_key" in call_args  # From additional context

    @pytest.mark.asyncio
    async def test_parse_json_success(self, mock_llm_factory, event_context):
        """Test successful JSON parsing."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)
        mock_model = mock_llm_factory.create_strands_model.return_value

        # Mock JSON response
        json_response = '{"action": "workflow", "device": "lights"}'
        mock_model.invoke.return_value = json_response

        result = await llm.parse_json("Parse this action")

        assert result == {"action": "workflow", "device": "lights"}

    @pytest.mark.asyncio
    async def test_parse_json_with_markdown(self, mock_llm_factory, event_context):
        """Test JSON parsing when response is wrapped in markdown."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)
        mock_model = mock_llm_factory.create_strands_model.return_value

        # Mock JSON response wrapped in markdown
        json_response = '```json\n{"action": "workflow", "device": "lights"}\n```'
        mock_model.invoke.return_value = json_response

        result = await llm.parse_json("Parse this action")

        assert result == {"action": "workflow", "device": "lights"}

    @pytest.mark.asyncio
    async def test_parse_json_failure(self, mock_llm_factory, event_context):
        """Test JSON parsing failure handling."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)
        mock_model = mock_llm_factory.create_strands_model.return_value

        # Mock invalid JSON response
        mock_model.invoke.return_value = "This is not JSON"

        result = await llm.parse_json("Parse this action")

        assert result == {}  # Should return empty dict on failure

    @pytest.mark.asyncio
    async def test_quick_decision_with_options(self, mock_llm_factory, event_context):
        """Test quick decision making with options."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)
        mock_model = mock_llm_factory.create_strands_model.return_value

        # Mock response that contains one of the options
        mock_model.invoke.return_value = "I think we should create a workflow for this"

        result = await llm.quick_decision(
            "Should this create a workflow?", ["workflow", "notification"]
        )

        assert result == "workflow"

    @pytest.mark.asyncio
    async def test_quick_decision_without_options(
        self, mock_llm_factory, event_context
    ):
        """Test quick decision making without predefined options."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)
        mock_model = mock_llm_factory.create_strands_model.return_value

        mock_model.invoke.return_value = "Custom response"

        result = await llm.quick_decision("What should we do?")

        assert result == "Custom response"

    def test_get_context_full(self, mock_llm_factory, event_context):
        """Test getting full execution context."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        context = llm.get_context()

        assert context == event_context["execution_context"]

    def test_get_context_specific_key(self, mock_llm_factory, event_context):
        """Test getting specific context key."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        user_id = llm.get_context("user_id")
        assert user_id == "U123456"

    def test_get_context_nested_key(self, mock_llm_factory, event_context):
        """Test getting nested context key with dot notation."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        room = llm.get_context("device_context.room")
        assert room == "bedroom"

    def test_get_context_missing_key(self, mock_llm_factory, event_context):
        """Test getting missing context key."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        missing = llm.get_context("nonexistent_key")
        assert missing is None

    def test_convenience_methods(self, mock_llm_factory, event_context):
        """Test convenience methods for common context access."""
        llm = EventHandlerLLM(mock_llm_factory, event_context)

        assert llm.get_original_request() == "turn on the lights"
        assert llm.get_timer_id() == "test123"
        assert llm.get_user_id() == "U123456"
        assert llm.get_channel() == "#general"


class TestEnhancedMessageBus:
    """Test enhanced MessageBus with dynamic event support."""

    def test_message_bus_initialization_with_registry(self):
        """Test MessageBus initializes with event registry."""
        message_bus = MessageBus()

        assert hasattr(message_bus, "event_registry")
        assert isinstance(message_bus.event_registry, MessageTypeRegistry)

    def test_publish_with_validation_success(self):
        """Test publishing with successful validation."""
        message_bus = MessageBus()
        message_bus.start()

        # Register event type with schema
        schema = {"timer_id": {"type": "string", "required": True}}
        message_bus.event_registry.register_event_type("TIMER_EXPIRED", "timer", schema)

        # Mock subscriber
        mock_callback = Mock()
        message_bus.subscribe("test_subscriber", "TIMER_EXPIRED", mock_callback)

        # Publish valid message
        valid_message = {"timer_id": "test123", "extra": "data"}
        message_bus.publish("timer", "TIMER_EXPIRED", valid_message)

        # Should call the callback (in a separate thread, so we need to wait)
        import time

        time.sleep(0.1)  # Allow thread to execute
        mock_callback.assert_called_once_with(valid_message)

    def test_publish_with_validation_failure(self):
        """Test publishing with validation failure (should still publish with warning)."""
        message_bus = MessageBus()
        message_bus.start()

        # Register event type with schema
        schema = {"timer_id": {"type": "string", "required": True}}
        message_bus.event_registry.register_event_type("TIMER_EXPIRED", "timer", schema)

        # Mock subscriber
        mock_callback = Mock()
        message_bus.subscribe("test_subscriber", "TIMER_EXPIRED", mock_callback)

        # Publish invalid message (missing required field)
        invalid_message = {"extra": "data"}

        with patch("common.message_bus.logger") as mock_logger:
            message_bus.publish("timer", "TIMER_EXPIRED", invalid_message)

            # Should log warning but still publish
            mock_logger.warning.assert_called()

            # Should still call the callback
            import time

            time.sleep(0.1)
            mock_callback.assert_called_once_with(invalid_message)

    def test_publish_unknown_event_type(self):
        """Test publishing unknown event type."""
        message_bus = MessageBus()
        message_bus.start()

        # Mock subscriber for unknown event
        mock_callback = Mock()
        message_bus.subscribe("test_subscriber", "UNKNOWN_EVENT", mock_callback)

        with patch("common.message_bus.logger") as mock_logger:
            message_bus.publish("unknown", "UNKNOWN_EVENT", {"data": "test"})

            # Should log warning about unknown event type
            mock_logger.warning.assert_called()

    def test_subscribe_to_dynamic_event_type(self):
        """Test subscribing to dynamic event types."""
        message_bus = MessageBus()

        mock_callback = Mock()
        message_bus.subscribe("test_subscriber", "CUSTOM_EVENT", mock_callback)

        # Should be able to subscribe to any event type
        assert "CUSTOM_EVENT" in message_bus._subscribers
        assert "test_subscriber" in message_bus._subscribers["CUSTOM_EVENT"]
        assert (
            mock_callback in message_bus._subscribers["CUSTOM_EVENT"]["test_subscriber"]
        )
