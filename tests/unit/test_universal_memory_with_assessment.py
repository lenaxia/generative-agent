"""Tests for UniversalMemory with assessment fields (summary, topics)."""

import time

import pytest

from common.providers.universal_memory_provider import UniversalMemory


class TestUniversalMemoryWithSummary:
    """Test UniversalMemory with summary field."""

    def test_create_memory_with_summary(self):
        """Test creating memory with summary field."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Full conversation content here",
            summary="User discussed project timeline",
            source_role="conversation",
            timestamp=time.time(),
        )

        assert memory.summary == "User discussed project timeline"

    def test_create_memory_without_summary(self):
        """Test creating memory without summary (optional)."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Content",
            source_role="conversation",
            timestamp=time.time(),
        )

        assert memory.summary is None

    def test_serialize_memory_with_summary(self):
        """Test serialization includes summary."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Content",
            summary="Brief summary",
            source_role="conversation",
            timestamp=time.time(),
        )

        data = memory.to_dict()
        assert "summary" in data
        assert data["summary"] == "Brief summary"

    def test_deserialize_memory_with_summary(self):
        """Test deserialization handles summary."""
        data = {
            "id": "test-id",
            "user_id": "user123",
            "memory_type": "conversation",
            "content": "Content",
            "summary": "Summary text",
            "source_role": "conversation",
            "timestamp": time.time(),
        }

        memory = UniversalMemory.from_dict(data)
        assert memory.summary == "Summary text"


class TestUniversalMemoryWithTopics:
    """Test UniversalMemory with topics field."""

    def test_create_memory_with_topics(self):
        """Test creating memory with topics field."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Content",
            source_role="conversation",
            timestamp=time.time(),
            topics=["Project Planning", "Budget Discussion"],
        )

        assert len(memory.topics) == 2
        assert "Project Planning" in memory.topics

    def test_create_memory_without_topics(self):
        """Test creating memory without topics (optional)."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Content",
            source_role="conversation",
            timestamp=time.time(),
        )

        assert memory.topics is None

    def test_serialize_memory_with_topics(self):
        """Test serialization includes topics."""
        memory = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Content",
            source_role="conversation",
            timestamp=time.time(),
            topics=["Topic 1", "Topic 2"],
        )

        data = memory.to_dict()
        assert "topics" in data
        assert data["topics"] == ["Topic 1", "Topic 2"]

    def test_deserialize_memory_with_topics(self):
        """Test deserialization handles topics."""
        data = {
            "id": "test-id",
            "user_id": "user123",
            "memory_type": "conversation",
            "content": "Content",
            "source_role": "conversation",
            "timestamp": time.time(),
            "topics": ["Topic A", "Topic B"],
        }

        memory = UniversalMemory.from_dict(data)
        assert len(memory.topics) == 2


class TestBackwardsCompatibility:
    """Test backwards compatibility with old memories."""

    def test_deserialize_old_memory_without_summary(self):
        """Test old memories without summary field."""
        data = {
            "id": "test-id",
            "user_id": "user123",
            "memory_type": "conversation",
            "content": "Content",
            "source_role": "conversation",
            "timestamp": time.time(),
            # No summary field
        }

        memory = UniversalMemory.from_dict(data)
        assert memory.summary is None

    def test_deserialize_old_memory_without_topics(self):
        """Test old memories without topics field."""
        data = {
            "id": "test-id",
            "user_id": "user123",
            "memory_type": "conversation",
            "content": "Content",
            "source_role": "conversation",
            "timestamp": time.time(),
            # No topics field
        }

        memory = UniversalMemory.from_dict(data)
        assert memory.topics is None or memory.topics == []

    def test_round_trip_with_all_fields(self):
        """Test round-trip serialization with all fields."""
        original = UniversalMemory(
            id="test-id",
            user_id="user123",
            memory_type="conversation",
            content="Full content",
            summary="Brief summary",
            source_role="conversation",
            timestamp=time.time(),
            importance=0.8,
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            topics=["Topic 1"],
            related_memories=["mem1"],
        )

        data = original.to_dict()
        restored = UniversalMemory.from_dict(data)

        assert restored.summary == original.summary
        assert restored.topics == original.topics
        assert restored.content == original.content
