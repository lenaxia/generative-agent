"""Tests for memory intents."""

import time

import pytest

from common.intents import MemoryWriteIntent


class TestMemoryWriteIntent:
    """Test MemoryWriteIntent validation and creation."""

    def test_memory_write_intent_valid(self):
        """Test creating a valid MemoryWriteIntent."""
        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="Test memory content",
            source_role="conversation",
            importance=0.7,
            metadata={"key": "value"},
            tags=["test"],
            related_memories=["mem1"],
        )

        assert intent.user_id == "user123"
        assert intent.memory_type == "conversation"
        assert intent.content == "Test memory content"
        assert intent.source_role == "conversation"
        assert intent.importance == 0.7
        assert intent.metadata == {"key": "value"}
        assert intent.tags == ["test"]
        assert intent.related_memories == ["mem1"]
        assert intent.validate() is True

    def test_memory_write_intent_minimal(self):
        """Test creating MemoryWriteIntent with minimal fields."""
        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="event",
            content="Event content",
            source_role="calendar",
        )

        assert intent.user_id == "user123"
        assert intent.memory_type == "event"
        assert intent.content == "Event content"
        assert intent.source_role == "calendar"
        assert intent.importance == 0.5  # Default
        assert intent.metadata is None
        assert intent.tags is None
        assert intent.related_memories is None
        assert intent.validate() is True

    def test_memory_write_intent_all_types(self):
        """Test all valid memory types."""
        valid_types = ["conversation", "event", "plan", "preference", "fact"]

        for memory_type in valid_types:
            intent = MemoryWriteIntent(
                user_id="user123",
                memory_type=memory_type,
                content="Content",
                source_role="test",
            )
            assert intent.validate() is True

    def test_memory_write_intent_invalid_type(self):
        """Test validation fails for invalid memory type."""
        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="invalid_type",
            content="Content",
            source_role="test",
        )

        assert intent.validate() is False

    def test_memory_write_intent_empty_user_id(self):
        """Test validation fails for empty user_id."""
        intent = MemoryWriteIntent(
            user_id="",
            memory_type="conversation",
            content="Content",
            source_role="test",
        )

        assert intent.validate() is False

    def test_memory_write_intent_empty_content(self):
        """Test validation fails for empty content."""
        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="",
            source_role="test",
        )

        assert intent.validate() is False

    def test_memory_write_intent_empty_source_role(self):
        """Test validation fails for empty source_role."""
        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="Content",
            source_role="",
        )

        assert intent.validate() is False

    def test_memory_write_intent_importance_bounds(self):
        """Test validation for importance bounds."""
        # Valid importance values
        for importance in [0.0, 0.5, 1.0]:
            intent = MemoryWriteIntent(
                user_id="user123",
                memory_type="conversation",
                content="Content",
                source_role="test",
                importance=importance,
            )
            assert intent.validate() is True

        # Invalid importance values
        for importance in [-0.1, 1.1, 2.0]:
            intent = MemoryWriteIntent(
                user_id="user123",
                memory_type="conversation",
                content="Content",
                source_role="test",
                importance=importance,
            )
            assert intent.validate() is False

    def test_memory_write_intent_serialization(self):
        """Test intent serialization to dict."""
        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="plan",
            content="Plan content",
            source_role="planning",
            importance=0.8,
            metadata={"workflow_id": "wf123"},
            tags=["project"],
            related_memories=["mem1", "mem2"],
        )

        data = intent.to_dict()

        assert data["type"] == "MemoryWriteIntent"
        assert "data" in data
        assert data["data"]["user_id"] == "user123"
        assert data["data"]["memory_type"] == "plan"
        assert data["data"]["content"] == "Plan content"
        assert data["data"]["source_role"] == "planning"
        assert data["data"]["importance"] == 0.8
        assert data["data"]["metadata"] == {"workflow_id": "wf123"}
        assert data["data"]["tags"] == ["project"]
        assert data["data"]["related_memories"] == ["mem1", "mem2"]

    def test_memory_write_intent_has_timestamp(self):
        """Test that intent has created_at timestamp."""
        before = time.time()
        intent = MemoryWriteIntent(
            user_id="user123",
            memory_type="conversation",
            content="Content",
            source_role="test",
        )
        after = time.time()

        assert hasattr(intent, "created_at")
        assert before <= intent.created_at <= after
