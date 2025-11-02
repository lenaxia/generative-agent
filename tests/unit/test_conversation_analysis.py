"""Tests for conversation analysis tool.

This module tests the analyze_conversation tool that triggers memory
importance assessment for unanalyzed messages in the realtime log.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.intents import MemoryWriteIntent
from common.memory_assessment import MemoryAssessment
from roles.shared_tools.conversation_analysis import analyze_conversation


@pytest.fixture
def mock_unanalyzed_messages():
    """Create mock unanalyzed messages."""
    return [
        {
            "id": "msg1",
            "user": "What's the weather like?",
            "assistant": "It's sunny and 72°F today.",
            "role": "conversation",
            "timestamp": time.time() - 100,
            "analyzed": False,
            "metadata": {},
        },
        {
            "id": "msg2",
            "user": "Should I bring an umbrella?",
            "assistant": "No need, there's no rain in the forecast.",
            "role": "conversation",
            "timestamp": time.time() - 50,
            "analyzed": False,
            "metadata": {},
        },
    ]


@pytest.fixture
def mock_assessment():
    """Create mock memory assessment."""
    return MemoryAssessment(
        importance=0.6,
        summary="User asked about weather and umbrella needs",
        tags=["weather", "conversation", "casual"],
        topics=["Weather Inquiry"],
        reasoning="Normal conversation about weather conditions",
    )


@pytest.mark.asyncio
async def test_analyze_conversation_success(mock_unanalyzed_messages, mock_assessment):
    """Test successful analysis creates memories."""
    with patch(
        "roles.shared_tools.conversation_analysis.get_unanalyzed_messages"
    ) as mock_get_unanalyzed, patch(
        "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
    ) as mock_assessor_class, patch(
        "roles.shared_tools.conversation_analysis.mark_as_analyzed"
    ) as mock_mark_analyzed, patch(
        "roles.shared_tools.conversation_analysis.get_intent_processor"
    ) as mock_get_processor:
        # Setup mocks
        mock_get_unanalyzed.return_value = mock_unanalyzed_messages
        mock_assessor = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=mock_assessment)
        mock_assessor.calculate_ttl = MagicMock(return_value=30 * 24 * 3600)
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        # Execute
        result = await analyze_conversation(user_id="test_user")

        # Verify
        assert result["success"] is True
        assert result["analyzed_count"] == 2
        assert result["memories_created"] == 2
        mock_mark_analyzed.assert_called_once()
        mock_processor.process_intents.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_conversation_no_unanalyzed():
    """Test handles no unanalyzed messages."""
    with patch(
        "roles.shared_tools.conversation_analysis.get_unanalyzed_messages"
    ) as mock_get_unanalyzed:
        mock_get_unanalyzed.return_value = []

        # Execute
        result = await analyze_conversation(user_id="test_user")

        # Verify
        assert result["success"] is True
        assert result["analyzed_count"] == 0
        assert result["memories_created"] == 0
        assert "No unanalyzed messages" in result["message"]


@pytest.mark.asyncio
async def test_analyze_conversation_marks_analyzed(
    mock_unanalyzed_messages, mock_assessment
):
    """Test messages marked as analyzed."""
    with patch(
        "roles.shared_tools.conversation_analysis.get_unanalyzed_messages"
    ) as mock_get_unanalyzed, patch(
        "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
    ) as mock_assessor_class, patch(
        "roles.shared_tools.conversation_analysis.mark_as_analyzed"
    ) as mock_mark_analyzed, patch(
        "roles.shared_tools.conversation_analysis.get_intent_processor"
    ) as mock_get_processor:
        # Setup mocks
        mock_get_unanalyzed.return_value = mock_unanalyzed_messages
        mock_assessor = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=mock_assessment)
        mock_assessor.calculate_ttl = MagicMock(return_value=30 * 24 * 3600)
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        # Execute
        await analyze_conversation(user_id="test_user")

        # Verify messages marked as analyzed
        mock_mark_analyzed.assert_called_once_with("test_user", ["msg1", "msg2"])


@pytest.mark.asyncio
async def test_analyze_conversation_graduated_ttl(
    mock_unanalyzed_messages, mock_assessment
):
    """Test correct TTL applied based on importance."""
    with patch(
        "roles.shared_tools.conversation_analysis.get_unanalyzed_messages"
    ) as mock_get_unanalyzed, patch(
        "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
    ) as mock_assessor_class, patch(
        "roles.shared_tools.conversation_analysis.mark_as_analyzed"
    ), patch(
        "roles.shared_tools.conversation_analysis.get_intent_processor"
    ) as mock_get_processor:
        # Setup mocks with high importance
        high_importance_assessment = MemoryAssessment(
            importance=0.8,
            summary="Important discussion",
            tags=["important"],
            topics=["Critical Topic"],
            reasoning="High importance conversation",
        )

        mock_get_unanalyzed.return_value = [mock_unanalyzed_messages[0]]
        mock_assessor = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=high_importance_assessment)
        mock_assessor.calculate_ttl = MagicMock(return_value=None)  # Permanent
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        # Execute
        await analyze_conversation(user_id="test_user")

        # Verify TTL calculation was called
        mock_assessor.calculate_ttl.assert_called_once_with(0.8)


@pytest.mark.asyncio
async def test_analyze_conversation_assessment_failure(mock_unanalyzed_messages):
    """Test handles assessment failures gracefully."""
    with patch(
        "roles.shared_tools.conversation_analysis.get_unanalyzed_messages"
    ) as mock_get_unanalyzed, patch(
        "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
    ) as mock_assessor_class, patch(
        "roles.shared_tools.conversation_analysis.mark_as_analyzed"
    ) as mock_mark_analyzed, patch(
        "roles.shared_tools.conversation_analysis.get_intent_processor"
    ) as mock_get_processor:
        # Setup mocks - assessment returns None (failure)
        mock_get_unanalyzed.return_value = mock_unanalyzed_messages
        mock_assessor = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=None)
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        # Execute
        result = await analyze_conversation(user_id="test_user")

        # Verify - should still succeed but create no memories
        assert result["success"] is True
        assert result["analyzed_count"] == 2
        assert result["memories_created"] == 0
        # Messages should still be marked as analyzed even if assessment fails
        mock_mark_analyzed.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_conversation_multiple_messages(mock_assessment):
    """Test analyzes multiple messages correctly."""
    messages = [
        {
            "id": f"msg{i}",
            "user": f"Question {i}",
            "assistant": f"Answer {i}",
            "role": "conversation",
            "timestamp": time.time() - (100 - i * 10),
            "analyzed": False,
            "metadata": {},
        }
        for i in range(5)
    ]

    with patch(
        "roles.shared_tools.conversation_analysis.get_unanalyzed_messages"
    ) as mock_get_unanalyzed, patch(
        "roles.shared_tools.conversation_analysis.MemoryImportanceAssessor"
    ) as mock_assessor_class, patch(
        "roles.shared_tools.conversation_analysis.mark_as_analyzed"
    ) as mock_mark_analyzed, patch(
        "roles.shared_tools.conversation_analysis.get_intent_processor"
    ) as mock_get_processor:
        # Setup mocks
        mock_get_unanalyzed.return_value = messages
        mock_assessor = AsyncMock()
        mock_assessor.assess_memory = AsyncMock(return_value=mock_assessment)
        mock_assessor.calculate_ttl = MagicMock(return_value=7 * 24 * 3600)
        mock_assessor_class.return_value = mock_assessor

        mock_processor = MagicMock()
        mock_processor.process_intents = AsyncMock()
        mock_get_processor.return_value = mock_processor

        # Execute
        result = await analyze_conversation(user_id="test_user")

        # Verify
        assert result["success"] is True
        assert result["analyzed_count"] == 5
        assert result["memories_created"] == 5
        assert mock_assessor.assess_memory.call_count == 5
        mock_mark_analyzed.assert_called_once_with(
            "test_user", [f"msg{i}" for i in range(5)]
        )
