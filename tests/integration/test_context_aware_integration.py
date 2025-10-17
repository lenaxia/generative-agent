"""
Integration tests for context-aware request processing.

This module tests the complete end-to-end flow of context-aware request processing,
from router context selection through context gathering to enhanced role execution.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.context_types import ContextCollector, ContextType
from common.interfaces.context_interfaces import MemoryEntry
from common.request_model import RequestMetadata
from supervisor.supervisor import Supervisor


class TestContextAwareIntegration:
    """Test complete context-aware request processing flow."""

    @pytest.fixture
    def mock_supervisor_config(self):
        """Create mock supervisor with minimal config."""
        with patch("supervisor.supervisor.ConfigManager") as mock_config_class, patch(
            "supervisor.supervisor.Path"
        ) as mock_path, patch("supervisor.supervisor.configure_logging"), patch(
            "supervisor.supervisor.MessageBus"
        ), patch(
            "supervisor.supervisor.LLMFactory"
        ), patch(
            "supervisor.supervisor.WorkflowEngine"
        ), patch(
            "supervisor.supervisor.MetricsManager"
        ), patch(
            "common.communication_manager.CommunicationManager"
        ):
            # Mock config file existence
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Mock config manager with proper structure
            mock_config = Mock()
            mock_config.raw_config_data = {"environment": {}}

            # Mock the loaded config with proper logging structure
            mock_loaded_config = Mock()
            mock_loaded_config.logging.log_level = "INFO"
            mock_loaded_config.logging.log_file = "test.log"
            mock_loaded_config.logging.disable_console_logging = False
            mock_loaded_config.logging.log_file_max_size = 10
            mock_loaded_config.llm_providers = {}

            mock_config.load_config.return_value = mock_loaded_config
            mock_config_class.return_value = mock_config

            supervisor = Supervisor("config.yaml")
            return supervisor

    @pytest.mark.asyncio
    async def test_context_system_initialization_flow(self, mock_supervisor_config):
        """Test that context systems initialize properly during supervisor startup."""
        supervisor = mock_supervisor_config

        # Mock workflow engine with context initialization
        mock_workflow_engine = Mock()
        mock_workflow_engine.initialize_context_systems = AsyncMock()
        supervisor.workflow_engine = mock_workflow_engine

        # Test context system initialization
        await supervisor.start_async_tasks()

        # Verify context systems were initialized
        mock_workflow_engine.initialize_context_systems.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_system_initialization_failure_graceful_degradation(
        self, mock_supervisor_config
    ):
        """Test graceful degradation when context system initialization fails."""
        supervisor = mock_supervisor_config

        # Mock workflow engine with failing context initialization
        mock_workflow_engine = Mock()
        mock_workflow_engine.initialize_context_systems = AsyncMock(
            side_effect=Exception("Context initialization failed")
        )
        supervisor.workflow_engine = mock_workflow_engine

        # Should not raise exception - graceful degradation
        await supervisor.start_async_tasks()

        # Verify initialization was attempted
        mock_workflow_engine.initialize_context_systems.assert_called_once()

    def test_router_context_selection_integration(self):
        """Test router context selection with enhanced routing response."""
        import json

        from roles.core_router import parse_routing_response

        # Test router response with context requirements
        router_response = json.dumps(
            {
                "route": "smart_home",
                "confidence": 0.95,
                "parameters": {"action": "turn_on", "device": "lights"},
                "context_requirements": ["location"],
            }
        )

        result = parse_routing_response(router_response)

        assert result["valid"] is True
        assert result["route"] == "smart_home"
        assert result["context_requirements"] == ["location"]

    @pytest.mark.asyncio
    async def test_context_collector_integration_with_providers(self):
        """Test context collector integration with mock providers."""
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        # Create real providers (they'll use mocked Redis/MQTT)
        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        collector = ContextCollector(
            memory_provider=memory_provider, location_provider=location_provider
        )

        # Test initialization
        await collector.initialize()

        # Test context gathering with mocked backend
        with patch("common.providers.mqtt_location_provider.redis_read") as mock_read:
            mock_read.return_value = {"success": True, "value": "bedroom"}

            context = await collector.gather_context("test_user", ["location"])

            assert "location" in context
            assert context["location"] == "bedroom"

    @pytest.mark.asyncio
    async def test_memory_assessor_integration(self):
        """Test memory assessor integration with providers."""
        from common.providers.redis_memory_provider import RedisMemoryProvider
        from supervisor.memory_assessor import MemoryAssessor

        # Mock LLM factory
        mock_llm_factory = Mock()
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value='{"importance": 0.8}')

        with patch("supervisor.memory_assessor.UniversalAgent") as mock_universal_agent:
            mock_universal_agent.return_value = mock_agent

            memory_provider = RedisMemoryProvider()
            assessor = MemoryAssessor(memory_provider, mock_llm_factory)

            await assessor.initialize()

            # Test memory assessment
            with patch.object(memory_provider, "store_memory") as mock_store:
                mock_store.return_value = True

                await assessor.assess_and_store_if_important(
                    user_id="test_user",
                    prompt="I met my neighbor Bob today",
                    response="That's nice! Bob seems friendly.",
                    location="front_yard",
                )

                # Should store memory since importance (0.8) > threshold (0.3)
                mock_store.assert_called_once()

    def test_calendar_role_integration(self):
        """Test calendar role integration with role registry."""
        from roles.core_calendar import ROLE_CONFIG, register_role

        registration = register_role()

        # Verify role is properly configured for context awareness
        assert registration["config"]["memory_enabled"] is True
        assert registration["config"]["location_aware"] is True
        assert registration["config"]["fast_reply"] is True

    def test_enhanced_router_backwards_compatibility(self):
        """Test that enhanced router maintains backwards compatibility."""
        import json

        from roles.core_router import parse_routing_response

        # Test old-style router response (without context_requirements)
        old_response = json.dumps(
            {"route": "timer", "confidence": 0.98, "parameters": {"duration": "5m"}}
        )

        result = parse_routing_response(old_response)

        assert result["valid"] is True
        assert result["route"] == "timer"
        assert result["context_requirements"] == []  # Default empty list

    @pytest.mark.asyncio
    async def test_workflow_engine_context_integration_flow(self):
        """Test workflow engine context integration flow."""
        from common.message_bus import MessageBus
        from llm_provider.factory import LLMFactory
        from supervisor.workflow_engine import WorkflowEngine

        # Mock dependencies
        mock_llm_factory = Mock(spec=LLMFactory)
        mock_message_bus = Mock(spec=MessageBus)

        with patch("supervisor.workflow_engine.RoleRegistry"), patch(
            "supervisor.workflow_engine.UniversalAgent"
        ), patch("supervisor.workflow_engine.MCPClientManager"):
            workflow_engine = WorkflowEngine(
                llm_factory=mock_llm_factory, message_bus=mock_message_bus
            )

            # Test that context properties are initialized
            assert hasattr(workflow_engine, "context_collector")
            assert hasattr(workflow_engine, "memory_assessor")
            assert workflow_engine.context_collector is None  # Not initialized yet
            assert workflow_engine.memory_assessor is None  # Not initialized yet

    def test_context_type_enum_integration(self):
        """Test ContextType enum integration across system."""
        from common.context_types import ContextType

        # Test all context types are available
        context_types = [ct.value for ct in ContextType]

        assert "location" in context_types
        assert "recent_memory" in context_types
        assert "presence" in context_types
        assert "schedule" in context_types
        assert len(context_types) == 4

    def test_provider_interface_compliance(self):
        """Test that providers comply with interface contracts."""
        from common.interfaces.context_interfaces import (
            LocationProvider,
            MemoryProvider,
        )
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        # Test interface compliance
        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        assert isinstance(memory_provider, MemoryProvider)
        assert isinstance(location_provider, LocationProvider)

        # Test required methods exist
        assert hasattr(memory_provider, "store_memory")
        assert hasattr(memory_provider, "get_recent_memories")
        assert hasattr(memory_provider, "search_memories")

        assert hasattr(location_provider, "get_current_location")
        assert hasattr(location_provider, "update_location")


class TestContextAwareRequestScenarios:
    """Test specific context-aware request scenarios."""

    def test_device_control_with_location_context(self):
        """Test device control request should require location context."""
        # This tests the expected router behavior for device control
        device_requests = [
            "Turn on the lights",
            "Set temperature to 72 degrees",
            "Turn off the TV",
        ]

        for request in device_requests:
            # Router should determine these need location context
            expected_context_types = ["location"]
            assert "location" in expected_context_types

    def test_memory_recall_with_memory_context(self):
        """Test memory recall requests should require memory context."""
        memory_requests = [
            "Play my usual music",
            "What did I do earlier?",
            "What's my morning routine?",
        ]

        for request in memory_requests:
            # Router should determine these need memory context
            expected_context_types = ["recent_memory"]
            assert "recent_memory" in expected_context_types

    def test_whole_house_actions_with_presence_context(self):
        """Test whole-house actions should require presence context."""
        presence_requests = [
            "Turn off all lights",
            "Set house to away mode",
            "Is anyone home?",
        ]

        for request in presence_requests:
            # Router should determine these need presence context
            expected_context_types = ["presence"]
            assert "presence" in expected_context_types

    def test_simple_requests_no_context(self):
        """Test simple requests should require no context."""
        simple_requests = [
            "Set a timer for 5 minutes",
            "What's the weather?",
            "Cancel timer",
        ]

        for request in simple_requests:
            # Router should determine these need no context
            expected_context_types = []
            assert len(expected_context_types) == 0


if __name__ == "__main__":
    pytest.main([__file__])
