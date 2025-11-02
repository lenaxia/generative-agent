"""Tests for MemoryWriteIntent processing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.intent_processor import IntentProcessor
from common.intents import MemoryWriteIntent


@pytest.fixture
def mock_memory_provider():
    """Mock UniversalMemoryProvider."""
    with patch(
        "common.providers.universal_memory_provider.UniversalMemoryProvider"
    ) as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def intent_processor():
    """Create IntentProcessor instance."""
    return IntentProcessor()


class TestMemoryWriteIntentProcessing:
    """Test MemoryWriteIntent processing."""

    @pytest.mark.asyncio
    async def test_memory_write_intent_processing(
        self, intent_processor, mock_memory_provider
    ):
        """Test MemoryWriteIntent is processed correctly."""
        mock_memory_provider.write_memory.return_value = "mem-123"

        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="Test memory",
            source_role="conversation",
            importance=0.7,
        )

        result = await intent_processor.process_intents([intent])

        assert result["processed"] == 1
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

        # Verify write_memory was called
        mock_memory_provider.write_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_write_intent_creates_memory(
        self, intent_processor, mock_memory_provider
    ):
        """Test intent creates memory in storage."""
        mock_memory_provider.write_memory.return_value = "mem-456"

        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="event",
            content="Event memory",
            source_role="calendar",
            importance=0.8,
            metadata={"event_id": "evt-123"},
            tags=["meeting"],
        )

        await intent_processor.process_intents([intent])

        # Verify correct parameters passed
        call_args = mock_memory_provider.write_memory.call_args
        assert call_args[1]["user_id"] == "user123"
        assert call_args[1]["memory_type"] == "event"
        assert call_args[1]["content"] == "Event memory"
        assert call_args[1]["source_role"] == "calendar"
        assert call_args[1]["importance"] == 0.8
        assert call_args[1]["metadata"] == {"event_id": "evt-123"}
        assert call_args[1]["tags"] == ["meeting"]

    @pytest.mark.asyncio
    async def test_memory_write_intent_with_metadata(
        self, intent_processor, mock_memory_provider
    ):
        """Test intent with metadata and tags."""
        mock_memory_provider.write_memory.return_value = "mem-789"

        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="plan",
            content="Plan result",
            source_role="planning",
            importance=0.6,
            metadata={"workflow_id": "wf-123", "task_count": 5},
            tags=["planning", "project"],
            related_memories=["mem-100", "mem-200"],
        )

        await intent_processor.process_intents([intent])

        call_args = mock_memory_provider.write_memory.call_args
        assert call_args[1]["metadata"] == {"workflow_id": "wf-123", "task_count": 5}
        assert call_args[1]["tags"] == ["planning", "project"]
        assert call_args[1]["related_memories"] == ["mem-100", "mem-200"]

    @pytest.mark.asyncio
    async def test_memory_write_intent_validation_failure(
        self, intent_processor, mock_memory_provider
    ):
        """Test invalid intent is not processed."""
        intent = MemoryWriteIntent(
            user_id="",  # Invalid: empty user_id
            memory_type="conversation",
            content="Test",
            source_role="test",
        )

        result = await intent_processor.process_intents([intent])

        assert result["processed"] == 0
        assert result["failed"] == 1
        assert len(result["errors"]) == 1

        # Verify write_memory was NOT called
        mock_memory_provider.write_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_write_intent_write_failure(
        self, intent_processor, mock_memory_provider
    ):
        """Test handling write failure."""
        mock_memory_provider.write_memory.return_value = None  # Write failed

        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="Test",
            source_role="test",
        )

        result = await intent_processor.process_intents([intent])

        # Intent processing should still succeed (fire-and-forget)
        assert result["processed"] == 1
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_multiple_memory_write_intents(
        self, intent_processor, mock_memory_provider
    ):
        """Test processing multiple memory write intents."""
        mock_memory_provider.write_memory.side_effect = ["mem-1", "mem-2", "mem-3"]

        intents = [
            MemoryWriteIntent(
                user_id="user123",
                memory_type="conversation",
                content=f"Memory {i}",
                source_role="conversation",
            )
            for i in range(3)
        ]

        result = await intent_processor.process_intents(intents)

        assert result["processed"] == 3
        assert result["failed"] == 0
        assert mock_memory_provider.write_memory.call_count == 3
