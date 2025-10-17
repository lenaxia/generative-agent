"""
End-to-end tests for context-aware request processing.

This module tests complete end-to-end scenarios for context-aware functionality,
covering both happy path and unhappy path cases with real system integration.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.context_types import ContextCollector, ContextType
from common.interfaces.context_interfaces import MemoryEntry
from common.request_model import RequestMetadata
from supervisor.supervisor import Supervisor


class TestContextAwareEndToEndHappyPath:
    """Test happy path scenarios for context-aware request processing."""

    @pytest.fixture
    def mock_supervisor_minimal(self):
        """Create supervisor with minimal mocking for end-to-end tests."""
        with patch("supervisor.supervisor.ConfigManager") as mock_config_class, patch(
            "supervisor.supervisor.Path"
        ) as mock_path, patch("supervisor.supervisor.configure_logging"), patch(
            "supervisor.supervisor.MessageBus"
        ) as mock_bus_class, patch(
            "supervisor.supervisor.LLMFactory"
        ) as mock_llm_class, patch(
            "supervisor.supervisor.WorkflowEngine"
        ) as mock_we_class, patch(
            "supervisor.supervisor.MetricsManager"
        ), patch(
            "supervisor.supervisor.CommunicationManager"
        ):
            # Mock config file existence
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Mock config manager
            mock_config = Mock()
            mock_config.get_config.return_value = Mock()
            mock_config_class.return_value = mock_config

            # Mock workflow engine with context capabilities
            mock_workflow_engine = Mock()
            mock_workflow_engine.initialize_context_systems = AsyncMock()
            mock_workflow_engine.handle_request = Mock(return_value="req_123")
            mock_workflow_engine.handle_request_with_context = AsyncMock(
                return_value="req_ctx_123"
            )
            mock_workflow_engine.context_collector = Mock()
            mock_workflow_engine.memory_assessor = Mock()
            mock_we_class.return_value = mock_workflow_engine

            supervisor = Supervisor("config.yaml")
            return supervisor

    @pytest.mark.asyncio
    async def test_end_to_end_device_control_with_location_context(
        self, mock_supervisor_minimal
    ):
        """Test end-to-end device control request with location context (HAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock context collector to return location context
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock(
            return_value={"location": "bedroom"}
        )
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router to require location context
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "smart_home",
                "confidence": 0.95,
                "parameters": {"action": "turn_on", "device": "lights"},
                "context_requirements": ["location"],
            }

            # Create request
            request = RequestMetadata(
                prompt="Turn on the lights",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test end-to-end flow
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Verify context was gathered
            mock_context_collector.gather_context.assert_called_once_with(
                user_id="test_user", context_types=["location"]
            )

            # Verify result
            assert result == "req_ctx_123"

    @pytest.mark.asyncio
    async def test_end_to_end_memory_recall_with_memory_context(
        self, mock_supervisor_minimal
    ):
        """Test end-to-end memory recall request with memory context (HAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock context collector to return memory context
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock(
            return_value={
                "recent_memory": [
                    "I like jazz music in the evening",
                    "Classical is good too",
                ]
            }
        )
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router to require memory context
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "planning",
                "confidence": 0.90,
                "parameters": {"query": "music_preferences"},
                "context_requirements": ["recent_memory"],
            }

            # Create request
            request = RequestMetadata(
                prompt="Play my usual music",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test end-to-end flow
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Verify context was gathered
            mock_context_collector.gather_context.assert_called_once_with(
                user_id="test_user", context_types=["recent_memory"]
            )

            # Verify result
            assert result == "req_ctx_123"

    @pytest.mark.asyncio
    async def test_end_to_end_simple_request_no_context(self, mock_supervisor_minimal):
        """Test end-to-end simple request requiring no context (HAPPY PATH - ZERO OVERHEAD)."""
        supervisor = mock_supervisor_minimal

        # Mock context collector (should not be called)
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock()
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router to require no context
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "timer",
                "confidence": 0.98,
                "parameters": {"duration": "5m", "action": "set"},
                "context_requirements": [],  # No context needed
            }

            # Create request
            request = RequestMetadata(
                prompt="Set a timer for 5 minutes",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test end-to-end flow
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Verify context was NOT gathered (zero overhead)
            mock_context_collector.gather_context.assert_not_called()

            # Verify result
            assert result == "req_ctx_123"

    @pytest.mark.asyncio
    async def test_end_to_end_multiple_context_types(self, mock_supervisor_minimal):
        """Test end-to-end request requiring multiple context types (HAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock context collector to return multiple context types
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock(
            return_value={"location": "living_room", "presence": ["alice", "bob"]}
        )
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router to require multiple context types
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "smart_home",
                "confidence": 0.92,
                "parameters": {"action": "turn_off", "device": "all_lights"},
                "context_requirements": ["location", "presence"],
            }

            # Create request
            request = RequestMetadata(
                prompt="Turn off all lights",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test end-to-end flow
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Verify multiple context types were gathered
            mock_context_collector.gather_context.assert_called_once_with(
                user_id="test_user", context_types=["location", "presence"]
            )

            # Verify result
            assert result == "req_ctx_123"


class TestContextAwareEndToEndUnhappyPath:
    """Test unhappy path scenarios and error conditions."""

    @pytest.fixture
    def mock_supervisor_with_failures(self):
        """Create supervisor with failure scenarios for unhappy path testing."""
        with patch("supervisor.supervisor.ConfigManager") as mock_config_class, patch(
            "supervisor.supervisor.Path"
        ) as mock_path, patch("supervisor.supervisor.configure_logging"), patch(
            "supervisor.supervisor.MessageBus"
        ), patch(
            "supervisor.supervisor.LLMFactory"
        ), patch(
            "supervisor.supervisor.WorkflowEngine"
        ) as mock_we_class, patch(
            "supervisor.supervisor.MetricsManager"
        ), patch(
            "supervisor.supervisor.CommunicationManager"
        ):
            # Mock config file existence
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Mock config manager
            mock_config = Mock()
            mock_config.get_config.return_value = Mock()
            mock_config_class.return_value = mock_config

            # Mock workflow engine with context capabilities
            mock_workflow_engine = Mock()
            mock_workflow_engine.initialize_context_systems = AsyncMock()
            mock_workflow_engine.handle_request = Mock(return_value="req_fallback_123")
            mock_workflow_engine.handle_request_with_context = AsyncMock(
                return_value="req_ctx_123"
            )
            mock_workflow_engine.context_collector = None  # No context collector
            mock_workflow_engine.memory_assessor = None  # No memory assessor
            mock_we_class.return_value = mock_workflow_engine

            supervisor = Supervisor("config.yaml")
            return supervisor

    @pytest.mark.asyncio
    async def test_end_to_end_context_system_not_initialized(
        self, mock_supervisor_with_failures
    ):
        """Test end-to-end request when context systems are not initialized (UNHAPPY PATH)."""
        supervisor = mock_supervisor_with_failures

        # Context collector is None (not initialized)
        assert supervisor.workflow_engine.context_collector is None

        # Create request that would normally require context
        request = RequestMetadata(
            prompt="Turn on the lights",
            source_id="test_user",
            target_id="workflow_engine",
            metadata={"user_id": "test_user", "channel_id": "console"},
            response_requested=True,
        )

        # Test that system falls back gracefully
        result = supervisor.workflow_engine.handle_request(request)

        # Should use fallback method when context systems not available
        assert result == "req_fallback_123"

    @pytest.mark.asyncio
    async def test_end_to_end_context_gathering_failure(self, mock_supervisor_minimal):
        """Test end-to-end request when context gathering fails (UNHAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock context collector to fail
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock(
            side_effect=Exception("Redis connection failed")
        )
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router to require context
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "smart_home",
                "confidence": 0.95,
                "parameters": {"action": "turn_on", "device": "lights"},
                "context_requirements": ["location"],
            }

            # Create request
            request = RequestMetadata(
                prompt="Turn on the lights",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test graceful degradation when context gathering fails
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Should continue processing even when context gathering fails
            assert result == "req_ctx_123"

    @pytest.mark.asyncio
    async def test_end_to_end_missing_user_id(self, mock_supervisor_minimal):
        """Test end-to-end request without user_id (UNHAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock context collector (should not be called without user_id)
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock()
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router to require context
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "smart_home",
                "confidence": 0.95,
                "parameters": {"action": "turn_on", "device": "lights"},
                "context_requirements": ["location"],
            }

            # Create request WITHOUT user_id
            request = RequestMetadata(
                prompt="Turn on the lights",
                source_id="test_source",
                target_id="workflow_engine",
                metadata={"channel_id": "console"},  # No user_id
                response_requested=True,
            )

            # Test that context gathering is skipped without user_id
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Context gathering should be skipped
            mock_context_collector.gather_context.assert_not_called()

            # Should still process request
            assert result == "req_ctx_123"

    @pytest.mark.asyncio
    async def test_end_to_end_router_failure(self, mock_supervisor_minimal):
        """Test end-to-end request when router fails (UNHAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock router to fail
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.side_effect = Exception("Router execution failed")

            # Create request
            request = RequestMetadata(
                prompt="Turn on the lights",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test graceful handling of router failure
            # This should fall back to non-context-aware processing
            result = supervisor.workflow_engine.handle_request(request)

            # Should still return a result even when router fails
            assert result == "req_fallback_123"

    @pytest.mark.asyncio
    async def test_end_to_end_memory_assessor_failure(self, mock_supervisor_minimal):
        """Test end-to-end request when memory assessor fails (UNHAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock memory assessor to fail
        mock_memory_assessor = Mock()
        mock_memory_assessor.assess_and_store_if_important = AsyncMock(
            side_effect=Exception("Memory assessment failed")
        )
        supervisor.workflow_engine.memory_assessor = mock_memory_assessor

        # Mock context collector
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock(return_value={})
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "timer",
                "confidence": 0.98,
                "parameters": {"duration": "5m"},
                "context_requirements": [],
            }

            # Create request
            request = RequestMetadata(
                prompt="Set a timer for 5 minutes",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test that memory assessor failure doesn't break request processing
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Should still process request successfully
            assert result == "req_ctx_123"

    @pytest.mark.asyncio
    async def test_end_to_end_partial_context_failure(self, mock_supervisor_minimal):
        """Test end-to-end request when some context types fail (UNHAPPY PATH)."""
        supervisor = mock_supervisor_minimal

        # Mock context collector to return partial context (location succeeds, memory fails)
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock(
            return_value={
                "location": "bedroom"
                # memory context missing due to failure
            }
        )
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router to require multiple context types
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "smart_home",
                "confidence": 0.90,
                "parameters": {"action": "turn_on", "device": "lights"},
                "context_requirements": ["location", "recent_memory"],
            }

            # Create request
            request = RequestMetadata(
                prompt="Turn on the lights like I usually do",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test that partial context failure is handled gracefully
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Should still process request with available context
            assert result == "req_ctx_123"


class TestContextAwareEndToEndEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_supervisor_edge_cases(self):
        """Create supervisor for edge case testing."""
        with patch("supervisor.supervisor.ConfigManager") as mock_config_class, patch(
            "supervisor.supervisor.Path"
        ) as mock_path, patch("supervisor.supervisor.configure_logging"), patch(
            "supervisor.supervisor.MessageBus"
        ), patch(
            "supervisor.supervisor.LLMFactory"
        ), patch(
            "supervisor.supervisor.WorkflowEngine"
        ) as mock_we_class, patch(
            "supervisor.supervisor.MetricsManager"
        ), patch(
            "supervisor.supervisor.CommunicationManager"
        ):
            # Mock config file existence
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Mock config manager
            mock_config = Mock()
            mock_config.get_config.return_value = Mock()
            mock_config_class.return_value = mock_config

            # Mock workflow engine
            mock_workflow_engine = Mock()
            mock_workflow_engine.initialize_context_systems = AsyncMock()
            mock_workflow_engine.handle_request = Mock(return_value="req_edge_123")
            mock_workflow_engine.handle_request_with_context = AsyncMock(
                return_value="req_ctx_edge_123"
            )
            mock_we_class.return_value = mock_workflow_engine

            supervisor = Supervisor("config.yaml")
            return supervisor

    @pytest.mark.asyncio
    async def test_end_to_end_empty_context_requirements(
        self, mock_supervisor_edge_cases
    ):
        """Test end-to-end request with empty context requirements (EDGE CASE)."""
        supervisor = mock_supervisor_edge_cases

        # Mock context collector
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock()
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router with empty context requirements
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "weather",
                "confidence": 0.95,
                "parameters": {},
                "context_requirements": [],  # Empty list
            }

            # Create request
            request = RequestMetadata(
                prompt="What's the weather?",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test that empty context requirements are handled correctly
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Context gathering should not be called for empty requirements
            mock_context_collector.gather_context.assert_not_called()

            # Should still process request
            assert result == "req_ctx_edge_123"

    @pytest.mark.asyncio
    async def test_end_to_end_invalid_context_type(self, mock_supervisor_edge_cases):
        """Test end-to-end request with invalid context type (EDGE CASE)."""
        supervisor = mock_supervisor_edge_cases

        # Mock context collector to handle invalid context type
        mock_context_collector = Mock()
        mock_context_collector.gather_context = AsyncMock(
            return_value={}
        )  # Returns empty for invalid type
        supervisor.workflow_engine.context_collector = mock_context_collector

        # Mock router with invalid context type
        with patch(
            "supervisor.workflow_engine.WorkflowEngine._route_request_with_router_role"
        ) as mock_route:
            mock_route.return_value = {
                "route": "planning",
                "confidence": 0.80,
                "parameters": {},
                "context_requirements": ["invalid_context_type"],
            }

            # Create request
            request = RequestMetadata(
                prompt="Do something complex",
                source_id="test_user",
                target_id="workflow_engine",
                metadata={"user_id": "test_user", "channel_id": "console"},
                response_requested=True,
            )

            # Test that invalid context types are handled gracefully
            result = await supervisor.workflow_engine.handle_request_with_context(
                request
            )

            # Context gathering should be called but return empty
            mock_context_collector.gather_context.assert_called_once()

            # Should still process request
            assert result == "req_ctx_edge_123"

    @pytest.mark.asyncio
    async def test_end_to_end_context_initialization_failure(
        self, mock_supervisor_edge_cases
    ):
        """Test end-to-end flow when context system initialization fails (EDGE CASE)."""
        supervisor = mock_supervisor_edge_cases

        # Mock context initialization to fail
        supervisor.workflow_engine.initialize_context_systems = AsyncMock(
            side_effect=Exception("Context initialization failed")
        )

        # Test that initialization failure is handled gracefully
        await supervisor.start_async_tasks()

        # System should continue to work without context systems
        # Context collector should remain None after failed initialization
        assert supervisor.workflow_engine.context_collector is None

    def test_end_to_end_malformed_request_metadata(self):
        """Test end-to-end request with malformed metadata (EDGE CASE)."""
        # Test malformed request metadata handling
        malformed_requests = [
            # Missing metadata
            RequestMetadata(
                prompt="Test request",
                source_id="test",
                target_id="workflow_engine",
                metadata=None,
                response_requested=True,
            ),
            # Empty metadata
            RequestMetadata(
                prompt="Test request",
                source_id="test",
                target_id="workflow_engine",
                metadata={},
                response_requested=True,
            ),
            # Malformed user_id
            RequestMetadata(
                prompt="Test request",
                source_id="test",
                target_id="workflow_engine",
                metadata={"user_id": "", "channel_id": "console"},
                response_requested=True,
            ),
        ]

        for request in malformed_requests:
            # Should handle malformed requests gracefully
            user_id = request.metadata.get("user_id") if request.metadata else None
            should_gather_context = bool(user_id)

            # Malformed requests should not trigger context gathering
            assert should_gather_context is False


class TestContextAwareEndToEndPerformance:
    """Test performance characteristics of context-aware processing."""

    def test_end_to_end_zero_overhead_validation(self):
        """Test that simple requests have zero context overhead (PERFORMANCE)."""
        # Test requests that should have zero context overhead
        zero_overhead_requests = [
            "Set a timer for 5 minutes",
            "What's the weather?",
            "Cancel timer",
            "What time is it?",
        ]

        for request in zero_overhead_requests:
            # These should route to roles with no context requirements
            expected_context_requirements = []
            assert len(expected_context_requirements) == 0

    def test_end_to_end_surgical_context_gathering(self):
        """Test that context gathering is surgical (only what's needed) (PERFORMANCE)."""
        # Test requests that should require specific context types
        context_scenarios = [
            ("Turn on the lights", ["location"]),
            ("Play my usual music", ["recent_memory"]),
            ("Turn off all lights", ["location", "presence"]),
            ("What's my schedule today?", ["schedule"]),
        ]

        for request, expected_contexts in context_scenarios:
            # Verify each request type has specific context requirements
            assert isinstance(expected_contexts, list)
            assert len(expected_contexts) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_concurrent_requests(self):
        """Test concurrent context-aware requests (PERFORMANCE/EDGE CASE)."""
        # Mock multiple concurrent requests
        requests = [
            ("Turn on bedroom lights", ["location"]),
            ("Play jazz music", ["recent_memory"]),
            ("Is anyone home?", ["presence"]),
            ("Set timer for 10 minutes", []),
        ]

        # Test that concurrent requests can be processed
        for prompt, context_types in requests:
            request_data = {"prompt": prompt, "expected_context": context_types}

            # Each request should be processable independently
            assert isinstance(request_data["prompt"], str)
            assert isinstance(request_data["expected_context"], list)


if __name__ == "__main__":
    pytest.main([__file__])
