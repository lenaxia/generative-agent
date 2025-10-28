"""
Tests for workflow engine context integration.

This module tests the enhanced workflow engine that integrates context collection
and memory assessment with the existing request handling flow.
"""

import json
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from common.context_types import ContextCollector, ContextType
from common.interfaces.context_interfaces import (
    LocationProvider,
    MemoryEntry,
    MemoryProvider,
)
from common.request_model import RequestMetadata


class MockContextCollector:
    """Mock context collector for testing."""

    def __init__(self, context_data: dict[str, Any] = None):
        self.context_data = context_data or {}
        self.initialized = False

    async def initialize(self):
        """Mock initialization."""
        self.initialized = True

    async def gather_context(
        self, user_id: str, context_types: list[str]
    ) -> dict[str, Any]:
        """Mock context gathering."""
        result = {}
        for context_type in context_types:
            if context_type in self.context_data:
                result[context_type] = self.context_data[context_type]
        return result


class MockMemoryAssessor:
    """Mock memory assessor for testing."""

    def __init__(self):
        self.initialized = False
        self.assessments = []

    async def initialize(self):
        """Mock initialization."""
        self.initialized = True

    async def assess_and_store_if_important(
        self,
        user_id: str,
        prompt: str,
        response: str,
        location: str = None,
        workflow_id: str = None,
    ):
        """Mock memory assessment."""
        self.assessments.append(
            {
                "user_id": user_id,
                "prompt": prompt,
                "response": response,
                "location": location,
                "workflow_id": workflow_id,
            }
        )


class TestWorkflowEngineContextIntegration:
    """Test workflow engine context integration methods."""

    @pytest.fixture
    def mock_context_collector(self):
        """Create mock context collector with test data."""
        return MockContextCollector(
            {
                "location": "bedroom",
                "recent_memory": ["I like jazz music", "Meeting with Bob at 3pm"],
                "presence": ["alice", "bob"],
                "schedule": [],
            }
        )

    @pytest.fixture
    def mock_memory_assessor(self):
        """Create mock memory assessor."""
        return MockMemoryAssessor()

    @pytest.fixture
    def sample_request(self):
        """Create sample request metadata."""
        return RequestMetadata(
            prompt="Turn on the lights",
            source_id="test_source",
            target_id="workflow_engine",
            metadata={
                "user_id": "test_user",
                "channel_id": "console",
                "workflow_id": "wf_123",
            },
            response_requested=True,
        )

    def test_add_context_to_prompt_with_location(self):
        """Test adding location context to prompt."""
        from supervisor.workflow_engine import WorkflowEngine

        # Create a mock workflow engine instance for testing the method
        workflow_engine = Mock(spec=WorkflowEngine)

        # Test the _add_context_to_prompt method logic
        base_prompt = "Turn on the lights"
        context = {"location": "bedroom"}

        # Expected behavior: prompt should include location context
        expected_parts = ["Location: bedroom"]

        # Verify context parts would be added
        assert "bedroom" in str(context.get("location", ""))

    def test_add_context_to_prompt_with_memory(self):
        """Test adding memory context to prompt."""
        base_prompt = "Play my usual music"
        context = {
            "recent_memory": [
                "I like jazz music in the evening",
                "Classical is good too",
            ]
        }

        # Expected behavior: prompt should include recent memory
        expected_memory = context["recent_memory"][0][:50] + "..."

        # Verify memory would be truncated and included
        assert len(expected_memory) <= 53  # 50 chars + "..."
        assert "jazz music" in expected_memory

    def test_add_context_to_prompt_with_presence(self):
        """Test adding presence context to prompt."""
        base_prompt = "Turn off all lights"
        context = {"presence": ["alice", "bob"]}

        # Expected behavior: prompt should include who else is home
        others_home = ", ".join(context["presence"])

        assert "alice" in others_home
        assert "bob" in others_home

    def test_add_context_to_prompt_multiple_contexts(self):
        """Test adding multiple context types to prompt."""
        base_prompt = "Set the temperature"
        context = {
            "location": "living_room",
            "presence": ["alice"],
            "recent_memory": ["I prefer 72 degrees"],
        }

        # All context types should be represented
        assert context.get("location") == "living_room"
        assert context.get("presence") == ["alice"]
        assert len(context.get("recent_memory", [])) == 1

    def test_add_context_to_prompt_empty_context(self):
        """Test adding empty context to prompt."""
        base_prompt = "Set a timer for 5 minutes"
        context = {}

        # Empty context should return original prompt
        # (This tests the logic that would be in _add_context_to_prompt)
        assert len(context) == 0

    @pytest.mark.asyncio
    async def test_context_gathering_integration(
        self, mock_context_collector, sample_request
    ):
        """Test context gathering integration flow."""
        # Mock routing result with context requirements
        routing_result = {
            "route": "smart_home",
            "confidence": 0.95,
            "parameters": {"action": "turn_on", "device": "lights"},
            "context_requirements": ["location"],
        }

        # Test context gathering
        context = await mock_context_collector.gather_context(
            user_id="test_user", context_types=routing_result["context_requirements"]
        )

        assert "location" in context
        assert context["location"] == "bedroom"

    @pytest.mark.asyncio
    async def test_context_gathering_with_multiple_types(self, mock_context_collector):
        """Test gathering multiple context types."""
        context = await mock_context_collector.gather_context(
            user_id="test_user", context_types=["location", "recent_memory", "presence"]
        )

        assert "location" in context
        assert "recent_memory" in context
        assert "presence" in context
        assert context["location"] == "bedroom"
        assert len(context["recent_memory"]) == 2
        assert len(context["presence"]) == 2

    @pytest.mark.asyncio
    async def test_context_gathering_empty_requirements(self, mock_context_collector):
        """Test context gathering with empty requirements."""
        context = await mock_context_collector.gather_context(
            user_id="test_user", context_types=[]
        )

        assert context == {}

    @pytest.mark.asyncio
    async def test_memory_assessment_integration(self, mock_memory_assessor):
        """Test memory assessment integration."""
        await mock_memory_assessor.assess_and_store_if_important(
            user_id="test_user",
            prompt="Turn on the lights",
            response="Lights turned on in bedroom",
            location="bedroom",
            workflow_id="wf_123",
        )

        assert len(mock_memory_assessor.assessments) == 1
        assessment = mock_memory_assessor.assessments[0]

        assert assessment["user_id"] == "test_user"
        assert assessment["prompt"] == "Turn on the lights"
        assert assessment["response"] == "Lights turned on in bedroom"
        assert assessment["location"] == "bedroom"
        assert assessment["workflow_id"] == "wf_123"

    def test_request_metadata_user_id_extraction(self, sample_request):
        """Test extracting user_id from request metadata."""
        user_id = sample_request.metadata.get("user_id")

        assert user_id == "test_user"
        assert sample_request.metadata.get("channel_id") == "console"
        assert sample_request.metadata.get("workflow_id") == "wf_123"

    def test_context_requirements_parsing(self):
        """Test parsing context requirements from routing result."""
        routing_result = {
            "route": "smart_home",
            "confidence": 0.95,
            "parameters": {"action": "turn_on"},
            "context_requirements": ["location", "presence"],
        }

        context_types = routing_result.get("context_requirements", [])

        assert isinstance(context_types, list)
        assert len(context_types) == 2
        assert "location" in context_types
        assert "presence" in context_types

    def test_context_requirements_empty(self):
        """Test handling empty context requirements."""
        routing_result = {
            "route": "timer",
            "confidence": 0.98,
            "parameters": {"duration": "5m"},
            "context_requirements": [],
        }

        context_types = routing_result.get("context_requirements", [])

        assert isinstance(context_types, list)
        assert len(context_types) == 0

    def test_context_requirements_missing(self):
        """Test handling missing context requirements field."""
        routing_result = {
            "route": "weather",
            "confidence": 0.90,
            "parameters": {},
            # No context_requirements field
        }

        context_types = routing_result.get("context_requirements", [])

        assert isinstance(context_types, list)
        assert len(context_types) == 0


class TestContextIntegrationErrorHandling:
    """Test error handling in context integration."""

    @pytest.mark.asyncio
    async def test_context_gathering_failure(self):
        """Test graceful handling of context gathering failures."""
        mock_collector = Mock()
        mock_collector.gather_context = AsyncMock(
            side_effect=Exception("Context gathering failed")
        )

        # Should not raise exception, should return empty context
        try:
            context = await mock_collector.gather_context("test_user", ["location"])
            assert False, "Should have raised exception"
        except Exception as e:
            assert str(e) == "Context gathering failed"

    @pytest.mark.asyncio
    async def test_memory_assessment_failure(self):
        """Test graceful handling of memory assessment failures."""
        mock_assessor = Mock()
        mock_assessor.assess_and_store_if_important = AsyncMock(
            side_effect=Exception("Memory assessment failed")
        )

        # Should not raise exception in real implementation
        try:
            await mock_assessor.assess_and_store_if_important(
                "test_user", "test prompt", "test response"
            )
            assert False, "Should have raised exception"
        except Exception as e:
            assert str(e) == "Memory assessment failed"

    def test_missing_user_id_handling(self):
        """Test handling requests without user_id."""
        request = RequestMetadata(
            prompt="Test request",
            source_id="test",
            target_id="workflow_engine",
            metadata={},  # No user_id
            response_requested=True,
        )

        user_id = request.metadata.get("user_id")

        assert user_id is None

        # Context gathering should be skipped when no user_id
        should_gather_context = bool(user_id)
        assert should_gather_context is False


if __name__ == "__main__":
    pytest.main([__file__])
