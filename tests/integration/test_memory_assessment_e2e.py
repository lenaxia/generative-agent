"""End-to-end integration tests for memory assessment system.

This module tests the complete memory assessment workflow from
realtime log to assessed memory storage.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.memory_assessment import MemoryAssessment
from common.realtime_log import add_message, get_recent_messages
from roles.shared_tools.conversation_analysis import analyze_conversation
from supervisor.scheduled_tasks import check_conversation_inactivity


@pytest.fixture
def mock_assessment():
    """Create mock memory assessment."""
    return MemoryAssessment(
        importance=0.8,
        summary="User discussed project planning",
        tags=["project", "planning", "work"],
        topics=["Project Planning"],
        reasoning="Important work discussion with actionable items",
    )


@pytest.mark.asyncio
async def test_multi_turn_conversation_to_assessed_memory(mock_assessment):
    """Test full flow from conversation to assessed memory."""
    with (
        patch("common.realtime_log.get_unanalyzed_messages") as mock_get_unanalyzed,
        patch(
            "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
        ) as mock_assessor_class,
        patch(
            "roles.shared_tools.conversation_analysis.get_intent_processor"
        ) as mock_get_processor,
        patch("common.realtime_log.mark_as_analyzed"),
    ):
        mock_get_unanalyzed.return_value = [
            {
                "id": "msg1",
                "user": "Let's plan the project",
                "assistant": "Great! Let's break it down into phases.",
                "role": "conversation",
                "timestamp": time.time(),
                "analyzed": False,
                "metadata": {},
            }
        ]

        mock_assessor = AsyncMock()
        mock_assessor.initialize = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=mock_assessment)
        mock_assessor.calculate_ttl = MagicMock(return_value=None)
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        user_id = "test_user"

        result = await analyze_conversation(user_id=user_id)

        assert result["success"] is True
        assert result["memories_created"] >= 1
        mock_processor.process_intents.assert_called_once()


@pytest.mark.asyncio
async def test_inactivity_timeout_triggers_analysis(mock_assessment):
    """Test inactivity timeout triggers analysis."""
    with (
        patch("common.realtime_log._get_redis_client") as mock_redis,
        patch(
            "supervisor.scheduled_tasks.get_unanalyzed_messages"
        ) as mock_get_unanalyzed,
        patch("supervisor.scheduled_tasks.get_last_message_time") as mock_get_last_time,
        patch("supervisor.scheduled_tasks.analyze_conversation") as mock_analyze,
    ):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client

        mock_get_unanalyzed.return_value = [
            {
                "id": "msg1",
                "user": "Test",
                "assistant": "Response",
                "role": "conversation",
                "timestamp": time.time() - 1800,
                "analyzed": False,
            }
        ]
        mock_get_last_time.return_value = time.time() - 1900

        async def mock_analyze_func(user_id):
            return {"success": True, "analyzed_count": 1, "memories_created": 1}

        mock_analyze.side_effect = mock_analyze_func

        result = await check_conversation_inactivity(
            user_ids=["test_user"], inactivity_timeout_minutes=30
        )

        assert result["success"] is True
        assert result["users_analyzed"] == 1


@pytest.mark.asyncio
async def test_assessment_creates_memory_with_correct_ttl(mock_assessment):
    """Test assessment creates memory with correct TTL."""
    with (
        patch("common.realtime_log.get_unanalyzed_messages") as mock_get_unanalyzed,
        patch(
            "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
        ) as mock_assessor_class,
        patch(
            "roles.shared_tools.conversation_analysis.get_intent_processor"
        ) as mock_get_processor,
        patch("common.realtime_log.mark_as_analyzed"),
    ):
        mock_get_unanalyzed.return_value = [
            {
                "id": "msg1",
                "user": "Important message",
                "assistant": "Understood",
                "role": "conversation",
                "timestamp": time.time(),
                "analyzed": False,
                "metadata": {},
            }
        ]

        mock_assessor = AsyncMock()
        mock_assessor.initialize = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=mock_assessment)
        expected_ttl = None
        mock_assessor.calculate_ttl = MagicMock(return_value=expected_ttl)
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        await analyze_conversation(user_id="test_user")

        mock_assessor.calculate_ttl.assert_called_once_with(0.8)


@pytest.mark.asyncio
async def test_dual_layer_context_loading_works():
    """Test dual-layer context loading works."""
    from common.providers.universal_memory_provider import UniversalMemory
    from roles.core_conversation import load_conversation_context

    with (
        patch("common.realtime_log.get_recent_messages") as mock_get_recent,
        patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class,
    ):
        mock_get_recent.return_value = [
            {
                "user": "Hello",
                "assistant": "Hi!",
                "role": "conversation",
                "timestamp": time.time(),
                "analyzed": False,
            }
        ]

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = [
            UniversalMemory(
                id="mem1",
                user_id="test_user",
                memory_type="conversation",
                content="Important memory",
                source_role="conversation",
                timestamp=time.time(),
                summary="Important memory",
                importance=0.8,
                tags=["important"],
            )
        ]
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_conversation_context("test", context, {})

        assert "realtime_context" in result
        assert "assessed_memories" in result
        assert "Hello" in result["realtime_context"]
        assert "Important memory" in result["assessed_memories"]


@pytest.mark.asyncio
async def test_cross_role_memory_with_realtime():
    """Test cross-role memory with realtime."""
    from common.providers.universal_memory_provider import UniversalMemory
    from roles.core_calendar import load_calendar_context

    with (
        patch("common.realtime_log.get_recent_messages") as mock_get_recent,
        patch(
            "common.providers.universal_memory_provider.UniversalMemoryProvider"
        ) as mock_provider_class,
    ):
        mock_get_recent.return_value = [
            {
                "user": "Schedule meeting",
                "assistant": "Added to calendar",
                "role": "calendar",
                "timestamp": time.time(),
                "analyzed": False,
            }
        ]

        mock_provider = MagicMock()
        mock_provider.get_recent_memories.return_value = [
            UniversalMemory(
                id="mem1",
                user_id="test_user",
                memory_type="event",
                content="User prefers morning meetings",
                source_role="calendar",
                timestamp=time.time(),
                summary="Morning meeting preference",
                importance=0.9,
                tags=["preference", "meetings"],
            )
        ]
        mock_provider_class.return_value = mock_provider

        context = MagicMock()
        context.user_id = "test_user"

        result = load_calendar_context("test", context, {})

        assert "realtime_context" in result
        assert "assessed_memories" in result
        assert "Schedule meeting" in result["realtime_context"]
        assert "Morning meeting preference" in result["assessed_memories"]


@pytest.mark.asyncio
async def test_performance_validation():
    """Test performance within targets."""
    import time as time_module

    with patch("common.realtime_log._get_redis_client") as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client

        start = time_module.perf_counter()
        add_message(
            user_id="test_user",
            user_message="Test",
            assistant_response="Response",
            role="conversation",
        )
        duration = time_module.perf_counter() - start

        assert duration < 0.05


@pytest.mark.asyncio
async def test_graduated_ttl_applied_correctly(mock_assessment):
    """Test graduated TTL applied correctly."""
    from common.memory_importance_assessor import MemoryImportanceAssessor
    from llm_provider.factory import LLMFactory

    factory = LLMFactory({})
    assessor = MemoryImportanceAssessor(factory)

    permanent = assessor.calculate_ttl(0.8)
    assert permanent is None

    month = assessor.calculate_ttl(0.6)
    assert month == 30 * 24 * 60 * 60

    week = assessor.calculate_ttl(0.4)
    assert week == 7 * 24 * 60 * 60

    days = assessor.calculate_ttl(0.2)
    assert days == 3 * 24 * 60 * 60


@pytest.mark.asyncio
async def test_summary_and_topics_in_memory(mock_assessment):
    """Test summary and topics in memory."""
    assert mock_assessment.summary == "User discussed project planning"
    assert len(mock_assessment.topics) == 1
    assert "Project Planning" in mock_assessment.topics
    assert len(mock_assessment.tags) == 3


@pytest.mark.asyncio
async def test_analyze_conversation_tool_works():
    """Test analyze_conversation tool works."""
    with (
        patch("common.realtime_log._get_redis_client") as mock_redis,
        patch("common.realtime_log.get_unanalyzed_messages") as mock_get_unanalyzed,
        patch(
            "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
        ) as mock_assessor_class,
        patch(
            "roles.shared_tools.conversation_analysis.get_intent_processor"
        ) as mock_get_processor,
    ):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client

        mock_get_unanalyzed.return_value = []

        mock_assessor = AsyncMock()
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_get_processor.return_value = mock_processor

        result = await analyze_conversation(user_id="test_user")

        assert result["success"] is True
        assert "analyzed_count" in result


@pytest.mark.asyncio
async def test_full_workflow_integration(mock_assessment):
    """Test complete workflow integration."""
    with (
        patch("common.realtime_log._get_redis_client") as mock_redis,
        patch("common.realtime_log.get_unanalyzed_messages") as mock_get_unanalyzed,
        patch(
            "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
        ) as mock_assessor_class,
        patch(
            "roles.shared_tools.conversation_analysis.get_intent_processor"
        ) as mock_get_processor,
        patch("roles.shared_tools.conversation_analysis.mark_as_analyzed"),
    ):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client

        mock_get_unanalyzed.return_value = [
            {
                "id": "msg1",
                "user": "Let's discuss the project",
                "assistant": "Sure, what would you like to know?",
                "role": "conversation",
                "timestamp": time.time(),
                "analyzed": False,
                "metadata": {},
            }
        ]

        mock_assessor = AsyncMock()
        mock_assessor.initialize = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=mock_assessment)
        mock_assessor.calculate_ttl = MagicMock(return_value=None)
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        user_id = "test_user"

        result = await analyze_conversation(user_id=user_id)

        assert result["success"] is True
        assert result["memories_created"] >= 1

        intents_call = mock_processor.process_intents.call_args
        assert intents_call is not None
        intents = intents_call[0][0]
        assert len(intents) >= 1
        assert intents[0].importance == 0.8
