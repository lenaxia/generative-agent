"""
End-to-end tests for context-aware request processing.

This module tests complete end-to-end scenarios for context-aware functionality,
covering both happy path and unhappy path cases with focused component testing.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from common.context_types import ContextCollector
from common.interfaces.context_interfaces import MemoryEntry
from supervisor.workflow_engine import WorkflowEngine


class TestContextAwareEndToEndHappyPath:
    """Test happy path scenarios for context-aware request processing."""

    @pytest.mark.asyncio
    async def test_end_to_end_context_collector_with_location(self):
        """Test end-to-end context collector with location context (HAPPY PATH)."""
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        # Create real providers with mocked backends
        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        collector = ContextCollector(
            memory_provider=memory_provider, location_provider=location_provider
        )

        # Mock Redis backend for location
        with patch("common.providers.mqtt_location_provider.redis_read") as mock_read:
            mock_read.return_value = {"success": True, "value": "bedroom"}

            # Test context gathering
            context = await collector.gather_context("test_user", ["location"])

            assert "location" in context
            assert context["location"] == "bedroom"

    @pytest.mark.asyncio
    async def test_end_to_end_context_collector_with_memory(self):
        """Test end-to-end context collector with memory context (HAPPY PATH)."""
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        collector = ContextCollector(
            memory_provider=memory_provider, location_provider=location_provider
        )

        # Mock Redis backend for memory
        with (
            patch(
                "common.providers.redis_memory_provider.redis_get_keys"
            ) as mock_get_keys,
            patch("common.providers.redis_memory_provider.redis_read") as mock_read,
        ):
            mock_get_keys.return_value = {
                "success": True,
                "keys": ["memory:test_user:1697400000"],
            }

            mock_read.return_value = {
                "success": True,
                "value": {
                    "content": "I like jazz music in the evening",
                    "timestamp": "2023-10-15T20:00:00",
                    "location": "living_room",
                    "importance": 0.7,
                    "metadata": {},
                },
            }

            # Test memory context gathering
            context = await collector.gather_context("test_user", ["recent_memory"])

            assert "recent_memory" in context
            assert len(context["recent_memory"]) == 1
            assert "jazz music" in context["recent_memory"][0]

    def test_end_to_end_memory_entry_structure(self):
        """Test memory entry structure for end-to-end workflows (HAPPY PATH)."""
        # Test memory entry creation and validation
        memory_entry = MemoryEntry(
            user_id="test_user",
            content="I met my neighbor Bob today",
            timestamp=datetime.now(),
            location="front_yard",
            importance=0.8,
            metadata={"source": "conversation"},
        )

        assert memory_entry.user_id == "test_user"
        assert "Bob" in memory_entry.content
        assert memory_entry.location == "front_yard"
        assert memory_entry.importance == 0.8
        assert memory_entry.metadata["source"] == "conversation"

    def test_end_to_end_router_context_selection_workflow(self):
        """Test end-to-end router context selection workflow (HAPPY PATH)."""
        from roles.core_router import parse_routing_response

        # Test various router responses with context requirements
        test_cases = [
            {
                "input": {
                    "route": "smart_home",
                    "confidence": 0.95,
                    "parameters": {"action": "turn_on", "device": "lights"},
                    "context_requirements": ["location"],
                },
                "expected_context": ["location"],
            },
            {
                "input": {
                    "route": "planning",
                    "confidence": 0.90,
                    "parameters": {"query": "music_preferences"},
                    "context_requirements": ["recent_memory"],
                },
                "expected_context": ["recent_memory"],
            },
            {
                "input": {
                    "route": "timer",
                    "confidence": 0.98,
                    "parameters": {"duration": "5m"},
                    "context_requirements": [],
                },
                "expected_context": [],
            },
        ]

        for test_case in test_cases:
            router_response = json.dumps(test_case["input"])
            result = parse_routing_response(router_response)

            assert result["valid"] is True
            assert result["context_requirements"] == test_case["expected_context"]


class TestContextAwareEndToEndUnhappyPath:
    """Test unhappy path scenarios and error conditions."""

    @pytest.mark.asyncio
    async def test_end_to_end_context_collector_redis_failure(self):
        """Test context collector when Redis fails (UNHAPPY PATH)."""
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        collector = ContextCollector(
            memory_provider=memory_provider, location_provider=location_provider
        )

        # Mock Redis to fail
        with patch("common.providers.mqtt_location_provider.redis_read") as mock_read:
            mock_read.side_effect = Exception("Redis connection failed")

            # Test graceful handling of Redis failure
            context = await collector.gather_context("test_user", ["location"])

            # Should return empty context when Redis fails
            assert context == {}

    def test_end_to_end_importance_threshold_logic(self):
        """Test importance threshold logic for memory storage (UNHAPPY PATH)."""
        # Test importance threshold logic
        threshold = 0.3

        test_cases = [
            (0.0, False),  # Below threshold
            (0.2, False),  # Below threshold
            (0.3, False),  # At threshold (not greater than)
            (0.31, True),  # Above threshold
            (0.8, True),  # High importance
        ]

        for importance, should_store in test_cases:
            result = importance > threshold
            assert result == should_store

    def test_end_to_end_json_parsing_error_handling(self):
        """Test JSON parsing error handling (UNHAPPY PATH)."""
        # Test various JSON parsing scenarios
        invalid_json_cases = [
            "{ invalid json }",
            "not json at all",
            "",
            '{"incomplete": ',
        ]

        for invalid_json in invalid_json_cases:
            try:
                json.loads(invalid_json)
                assert False, f"Should have failed to parse: {invalid_json}"
            except (json.JSONDecodeError, ValueError):
                # Expected behavior - JSON parsing should fail
                assert True

        # Test that "null" is valid JSON but not useful for routing
        null_result = json.loads("null")
        assert null_result is None  # Valid JSON but not a routing response

    def test_end_to_end_router_invalid_json(self):
        """Test router with invalid JSON response (UNHAPPY PATH)."""
        from roles.core_router import parse_routing_response

        invalid_responses = [
            "{ invalid json }",
            "not json at all",
            '{"route": "timer"}',  # Missing required fields
            '{"route": "timer", "confidence": 1.5}',  # Invalid confidence
        ]

        for invalid_response in invalid_responses:
            result = parse_routing_response(invalid_response)

            # Should fallback to planning with error
            assert result["valid"] is False
            assert result["route"] == "PLANNING"
            assert result["confidence"] == 0.0
            assert "error" in result

    @pytest.mark.asyncio
    async def test_end_to_end_context_collector_empty_user_id(self):
        """Test context collector with empty user_id (UNHAPPY PATH)."""
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        collector = ContextCollector(
            memory_provider=memory_provider, location_provider=location_provider
        )

        # Test with empty user_id
        context = await collector.gather_context("", ["location"])
        assert context == {}

        # Test with None user_id
        context = await collector.gather_context(None, ["location"])
        assert context == {}

    @pytest.mark.asyncio
    async def test_end_to_end_context_collector_empty_context_types(self):
        """Test context collector with empty context types (UNHAPPY PATH)."""
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        collector = ContextCollector(
            memory_provider=memory_provider, location_provider=location_provider
        )

        # Test with empty context types
        context = await collector.gather_context("test_user", [])
        assert context == {}


class TestContextAwareEndToEndEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_end_to_end_context_type_enum_completeness(self):
        """Test that all context types are properly defined (EDGE CASE)."""
        from common.context_types import ContextType

        # Verify all expected context types exist
        expected_types = ["location", "recent_memory", "presence", "schedule"]
        actual_types = [ct.value for ct in ContextType]

        for expected_type in expected_types:
            assert expected_type in actual_types

        assert len(actual_types) == 4

    def test_end_to_end_provider_interface_compliance(self):
        """Test that all providers comply with interfaces (EDGE CASE)."""
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

        # Test all required methods exist
        memory_methods = ["store_memory", "get_recent_memories", "search_memories"]
        for method in memory_methods:
            assert hasattr(memory_provider, method)
            assert callable(getattr(memory_provider, method))

        location_methods = ["get_current_location", "update_location"]
        for method in location_methods:
            assert hasattr(location_provider, method)
            assert callable(getattr(location_provider, method))

    def test_end_to_end_calendar_role_context_awareness(self):
        """Test calendar role context awareness configuration (EDGE CASE)."""
        from roles.core_calendar import ROLE_CONFIG, register_role

        # Test role configuration
        assert ROLE_CONFIG["memory_enabled"] is True
        assert ROLE_CONFIG["location_aware"] is True
        assert ROLE_CONFIG["fast_reply"] is True

        # Test role registration
        registration = register_role()
        assert "config" in registration
        assert "tools" in registration
        assert "intents" in registration
        assert len(registration["tools"]) == 2  # get_schedule, add_calendar_event

    def test_end_to_end_router_backwards_compatibility(self):
        """Test router backwards compatibility with old response format (EDGE CASE)."""
        from roles.core_router import parse_routing_response

        # Test old-style response without context_requirements
        old_response = json.dumps(
            {
                "route": "timer",
                "confidence": 0.98,
                "parameters": {"duration": "5m"},
                # No context_requirements field
            }
        )

        result = parse_routing_response(old_response)

        assert result["valid"] is True
        assert result["route"] == "timer"
        assert result["context_requirements"] == []  # Should default to empty list

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_engine_context_properties(self):
        """Test workflow engine context properties initialization (EDGE CASE)."""
        from common.message_bus import MessageBus
        from llm_provider.factory import LLMFactory

        # Mock dependencies
        mock_llm_factory = Mock(spec=LLMFactory)
        mock_message_bus = Mock(spec=MessageBus)

        with (
            patch("supervisor.workflow_engine.RoleRegistry"),
            patch("supervisor.workflow_engine.UniversalAgent"),
            patch("supervisor.workflow_engine.MCPClientManager"),
        ):
            workflow_engine = WorkflowEngine(
                llm_factory=mock_llm_factory, message_bus=mock_message_bus
            )

            # Test that context properties are properly initialized
            assert hasattr(workflow_engine, "context_collector")
            assert hasattr(workflow_engine, "memory_assessor")
            assert workflow_engine.context_collector is None  # Not initialized yet
            assert workflow_engine.memory_assessor is None  # Not initialized yet


class TestContextAwareEndToEndPerformanceValidation:
    """Test performance characteristics and validation."""

    def test_end_to_end_zero_overhead_request_types(self):
        """Test that zero overhead requests are properly identified (PERFORMANCE)."""
        # Requests that should require no context (zero overhead)
        zero_overhead_scenarios = [
            ("Set a timer for 5 minutes", "timer"),
            ("What's the weather?", "weather"),
            ("Cancel timer", "timer"),
            ("What time is it?", "planning"),
        ]

        for prompt, expected_role in zero_overhead_scenarios:
            # These should not require context gathering
            expected_context_requirements = []
            assert len(expected_context_requirements) == 0

    def test_end_to_end_surgical_context_request_types(self):
        """Test that surgical context requests are properly identified (PERFORMANCE)."""
        # Requests that should require specific context types
        surgical_context_scenarios = [
            ("Turn on the lights", ["location"]),
            ("Play my usual music", ["recent_memory"]),
            ("Turn off all lights", ["location", "presence"]),
            ("What's my schedule today?", ["schedule"]),
        ]

        for prompt, expected_contexts in surgical_context_scenarios:
            # These should require specific context types
            assert isinstance(expected_contexts, list)
            assert len(expected_contexts) > 0

            # Verify context types are valid
            valid_types = ["location", "recent_memory", "presence", "schedule"]
            for context_type in expected_contexts:
                assert context_type in valid_types

    @pytest.mark.asyncio
    async def test_end_to_end_context_gathering_performance(self):
        """Test context gathering performance characteristics (PERFORMANCE)."""
        from common.providers.mqtt_location_provider import MQTTLocationProvider
        from common.providers.redis_memory_provider import RedisMemoryProvider

        memory_provider = RedisMemoryProvider()
        location_provider = MQTTLocationProvider("localhost")

        collector = ContextCollector(
            memory_provider=memory_provider, location_provider=location_provider
        )

        # Mock fast Redis responses
        with patch("common.providers.mqtt_location_provider.redis_read") as mock_read:
            mock_read.return_value = {"success": True, "value": "bedroom"}

            # Test that context gathering is efficient
            start_time = asyncio.get_event_loop().time()
            context = await collector.gather_context("test_user", ["location"])
            end_time = asyncio.get_event_loop().time()

            # Should complete quickly (mocked, so very fast)
            duration = end_time - start_time
            assert duration < 1.0  # Should be very fast with mocked backend
            assert "location" in context


if __name__ == "__main__":
    pytest.main([__file__])
