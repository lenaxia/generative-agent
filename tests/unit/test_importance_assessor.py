"""Tests for MemoryImportanceAssessor."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.memory_assessment import MemoryAssessment
from common.memory_importance_assessor import MemoryImportanceAssessor
from llm_provider.factory import LLMFactory, LLMType


@pytest.fixture
def mock_llm_factory():
    """Mock LLMFactory."""
    factory = MagicMock(spec=LLMFactory)
    mock_agent = AsyncMock()
    factory.get_agent.return_value = mock_agent
    return factory, mock_agent


@pytest.fixture
def assessor(mock_llm_factory):
    """Create assessor instance."""
    factory, _ = mock_llm_factory
    return MemoryImportanceAssessor(factory)


class TestImportanceAssessorInitialization:
    """Test assessor initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_weak_model(self, mock_llm_factory):
        """Test assessor initializes with WEAK model."""
        factory, mock_agent = mock_llm_factory

        assessor = MemoryImportanceAssessor(factory)
        await assessor.initialize()

        factory.get_agent.assert_called_once_with(LLMType.WEAK)
        assert assessor.agent is not None


class TestMemoryAssessment:
    """Test memory assessment functionality."""

    @pytest.mark.asyncio
    async def test_assess_single_turn(self, assessor, mock_llm_factory):
        """Test assessing a single turn conversation."""
        await assessor.initialize()
        _, mock_agent = mock_llm_factory

        # Mock LLM response
        assessment_json = json.dumps(
            {
                "importance": 0.8,
                "summary": "User scheduled team meeting for tomorrow",
                "tags": ["meeting", "work", "team"],
                "topics": ["Team Meetings"],
                "reasoning": "Important work event with specific time",
            }
        )
        mock_agent.execute.return_value = assessment_json

        result = await assessor.assess_memory(
            user_message="Can you schedule a team meeting for tomorrow?",
            assistant_response="I've scheduled the team meeting for tomorrow at 2pm.",
            source_role="calendar",
        )

        assert result is not None
        assert isinstance(result, MemoryAssessment)
        assert result.importance == 0.8
        assert result.summary == "User scheduled team meeting for tomorrow"
        assert len(result.tags) == 3
        assert len(result.topics) == 1

    @pytest.mark.asyncio
    async def test_assess_with_context(self, assessor, mock_llm_factory):
        """Test assessment with additional context."""
        await assessor.initialize()
        _, mock_agent = mock_llm_factory

        assessment_json = json.dumps(
            {
                "importance": 0.9,
                "summary": "User set preference for morning meetings",
                "tags": ["preference", "meetings", "morning"],
                "topics": ["Meeting Preferences"],
                "reasoning": "Explicit user preference to remember",
            }
        )
        mock_agent.execute.return_value = assessment_json

        result = await assessor.assess_memory(
            user_message="I prefer morning meetings",
            assistant_response="I'll remember that you prefer morning meetings.",
            source_role="conversation",
            context={"location": "home", "time": "evening"},
        )

        assert result is not None
        assert result.importance == 0.9

    @pytest.mark.asyncio
    async def test_assess_below_threshold_returns_result(
        self, assessor, mock_llm_factory
    ):
        """Test assessment below threshold still returns result."""
        await assessor.initialize()
        _, mock_agent = mock_llm_factory

        assessment_json = json.dumps(
            {
                "importance": 0.3,
                "summary": "Casual greeting exchange",
                "tags": ["greeting"],
                "topics": [],
                "reasoning": "Simple greeting, low importance",
            }
        )
        mock_agent.execute.return_value = assessment_json

        result = await assessor.assess_memory(
            user_message="Hi there!",
            assistant_response="Hello! How can I help you today?",
            source_role="conversation",
        )

        # Should still return result even if below threshold
        assert result is not None
        assert result.importance == 0.3

    @pytest.mark.asyncio
    async def test_assess_timeout(self, assessor, mock_llm_factory):
        """Test handling assessment timeout."""
        await assessor.initialize()
        _, mock_agent = mock_llm_factory

        # Mock timeout
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return "{}"

        mock_agent.execute = slow_execute

        result = await assessor.assess_memory(
            user_message="Test",
            assistant_response="Response",
            source_role="conversation",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_assess_json_parse_failure(self, assessor, mock_llm_factory):
        """Test handling JSON parse failure."""
        await assessor.initialize()
        _, mock_agent = mock_llm_factory

        # Mock invalid JSON
        mock_agent.execute.return_value = "This is not JSON"

        result = await assessor.assess_memory(
            user_message="Test",
            assistant_response="Response",
            source_role="conversation",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_assess_validation_failure(self, assessor, mock_llm_factory):
        """Test handling Pydantic validation failure."""
        await assessor.initialize()
        _, mock_agent = mock_llm_factory

        # Mock invalid data (importance out of bounds)
        assessment_json = json.dumps(
            {
                "importance": 1.5,  # Invalid
                "summary": "Test",
                "tags": ["test"],
                "topics": [],
                "reasoning": "Test",
            }
        )
        mock_agent.execute.return_value = assessment_json

        result = await assessor.assess_memory(
            user_message="Test",
            assistant_response="Response",
            source_role="conversation",
        )

        assert result is None


class TestTTLCalculation:
    """Test TTL calculation logic."""

    def test_ttl_permanent_for_high_importance(self):
        """Test permanent storage for importance >= 0.7."""
        assessor = MemoryImportanceAssessor(MagicMock())

        assert assessor.calculate_ttl(0.7) is None
        assert assessor.calculate_ttl(0.8) is None
        assert assessor.calculate_ttl(0.9) is None
        assert assessor.calculate_ttl(1.0) is None

    def test_ttl_one_month_for_medium(self):
        """Test 30 days for importance 0.5-0.7."""
        assessor = MemoryImportanceAssessor(MagicMock())

        ttl = assessor.calculate_ttl(0.5)
        assert ttl == 30 * 24 * 60 * 60

        ttl = assessor.calculate_ttl(0.6)
        assert ttl == 30 * 24 * 60 * 60

    def test_ttl_one_week_for_low(self):
        """Test 7 days for importance 0.3-0.5."""
        assessor = MemoryImportanceAssessor(MagicMock())

        ttl = assessor.calculate_ttl(0.3)
        assert ttl == 7 * 24 * 60 * 60

        ttl = assessor.calculate_ttl(0.4)
        assert ttl == 7 * 24 * 60 * 60

    def test_ttl_three_days_for_very_low(self):
        """Test 3 days for importance < 0.3."""
        assessor = MemoryImportanceAssessor(MagicMock())

        ttl = assessor.calculate_ttl(0.0)
        assert ttl == 3 * 24 * 60 * 60

        ttl = assessor.calculate_ttl(0.2)
        assert ttl == 3 * 24 * 60 * 60
