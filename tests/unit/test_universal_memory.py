"""Tests for UniversalMemory data model."""

import time
from datetime import datetime

import pytest

from common.providers.universal_memory_provider import UniversalMemory


class TestUniversalMemoryCreation:
    """Test creating universal memory instances."""

    def test_universal_memory_creation_with_all_fields(self):
        """Test creating a universal memory with all fields."""
        memory = UniversalMemory(
            id="test-id-123",
            user_id="user123",
            memory_type="conversation",
            content="Test memory content",
            source_role="conversation",
            timestamp=time.time(),
            importance=0.7,
            metadata={"key": "value"},
            tags=["test", "memory"],
            related_memories=["mem-456"],
        )

        assert memory.id == "test-id-123"
        assert memory.user_id == "user123"
        assert memory.memory_type == "conversation"
        assert memory.content == "Test memory content"
        assert memory.source_role == "conversation"
        assert memory.importance == 0.7
        assert memory.metadata == {"key": "value"}
        assert memory.tags == ["test", "memory"]
        assert memory.related_memories == ["mem-456"]

    def test_universal_memory_creation_with_minimal_fields(self):
        """Test creating memory with only required fields."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="event",
            content="Event content",
            source_role="calendar",
            timestamp=time.time(),
        )

        assert memory.id == "test-id"
        assert memory.importance == 0.5  # Default
        assert memory.metadata is None
        assert memory.tags is None
        assert memory.related_memories is None

    def test_universal_memory_defaults(self):
        """Test default values are applied correctly."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="plan",
            content="Plan content",
            source_role="planning",
            timestamp=time.time(),
        )

        assert memory.importance == 0.5
        assert memory.metadata is None
        assert memory.tags is None
        assert memory.related_memories is None


class TestUniversalMemorySerialization:
    """Test memory serialization and deserialization."""

    def test_to_dict_with_all_fields(self):
        """Test converting memory to dictionary."""
        timestamp = time.time()
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Test content",
            source_role="conversation",
            timestamp=timestamp,
            importance=0.8,
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            related_memories=["mem1", "mem2"],
        )

        result = memory.to_dict()

        assert result["id"] == "test-id"
        assert result["user_id"] == "user123"
        assert result["memory_type"] == "conversation"
        assert result["content"] == "Test content"
        assert result["source_role"] == "conversation"
        assert result["timestamp"] == timestamp
        assert result["importance"] == 0.8
        assert result["metadata"] == {"key": "value"}
        assert result["tags"] == ["tag1", "tag2"]
        assert result["related_memories"] == ["mem1", "mem2"]

    def test_to_dict_with_none_values(self):
        """Test to_dict handles None values correctly."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="event",
            content="Event",
            source_role="calendar",
            timestamp=time.time(),
        )

        result = memory.to_dict()

        assert result["metadata"] == {}
        assert result["tags"] == []
        assert result["related_memories"] == []

    def test_from_dict_with_all_fields(self):
        """Test creating memory from dictionary."""
        timestamp = time.time()
        data = {
            "id": "test-id",
            "user_id": "user123",
            "memory_type": "plan",
            "content": "Plan content",
            "source_role": "planning",
            "timestamp": timestamp,
            "importance": 0.9,
            "metadata": {"workflow_id": "wf-123"},
            "tags": ["planning", "project"],
            "related_memories": ["mem-abc"],
        }

        memory = UniversalMemory.from_dict(data)

        assert memory.id == "test-id"
        assert memory.user_id == "user123"
        assert memory.memory_type == "plan"
        assert memory.content == "Plan content"
        assert memory.source_role == "planning"
        assert memory.timestamp == timestamp
        assert memory.importance == 0.9
        assert memory.metadata == {"workflow_id": "wf-123"}
        assert memory.tags == ["planning", "project"]
        assert memory.related_memories == ["mem-abc"]

    def test_from_dict_with_missing_optional_fields(self):
        """Test from_dict handles missing optional fields."""
        data = {
            "id": "test-id",
            "user_id": "user123",
            "memory_type": "fact",
            "content": "Fact content",
            "source_role": "conversation",
            "timestamp": time.time(),
        }

        memory = UniversalMemory.from_dict(data)

        assert memory.importance == 0.5  # Default
        assert memory.metadata == {}
        assert memory.tags == []
        assert memory.related_memories == []

    def test_round_trip_serialization(self):
        """Test memory survives round-trip serialization."""
        original = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="preference",
            content="User prefers dark mode",
            source_role="conversation",
            timestamp=time.time(),
            importance=0.6,
            metadata={"setting": "theme"},
            tags=["preference", "ui"],
            related_memories=[],
        )

        # Round trip
        data = original.to_dict()
        restored = UniversalMemory.from_dict(data)

        assert restored.id == original.id
        assert restored.user_id == original.user_id
        assert restored.memory_type == original.memory_type
        assert restored.content == original.content
        assert restored.source_role == original.source_role
        assert restored.timestamp == original.timestamp
        assert restored.importance == original.importance
        assert restored.metadata == original.metadata
        assert restored.tags == original.tags
        assert restored.related_memories == original.related_memories


class TestUniversalMemoryValidation:
    """Test memory validation rules."""

    def test_valid_memory_types(self):
        """Test all valid memory types are accepted."""
        valid_types = ["conversation", "event", "plan", "preference", "fact"]

        for memory_type in valid_types:
            memory = UniversalMemory(
                id="test-id",
                user_id="user123",
                memory_type=memory_type,
                content="Content",
                source_role="test",
                timestamp=time.time(),
            )
            assert memory.memory_type == memory_type

    def test_importance_bounds(self):
        """Test importance values are within valid range."""
        # Valid importance values
        for importance in [0.0, 0.5, 1.0]:
            memory = UniversalMemory(
                id="test-id",
                user_id="user123",
                memory_type="conversation",
                content="Content",
                source_role="test",
                timestamp=time.time(),
                importance=importance,
            )
            assert memory.importance == importance

    def test_empty_content_allowed(self):
        """Test empty content is allowed (validation happens at provider level)."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="",
            source_role="test",
            timestamp=time.time(),
        )
        assert memory.content == ""

    def test_empty_lists_allowed(self):
        """Test empty lists are valid."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Content",
            source_role="test",
            timestamp=time.time(),
            tags=[],
            related_memories=[],
        )
        assert memory.tags == []
        assert memory.related_memories == []
