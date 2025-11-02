"""Tests for UniversalMemoryProvider."""

import time
from unittest.mock import MagicMock, patch

import pytest

from common.providers.universal_memory_provider import (
    UniversalMemory,
    UniversalMemoryProvider,
)


@pytest.fixture
def memory_provider():
    """Create a memory provider instance for testing."""
    return UniversalMemoryProvider()


@pytest.fixture
def mock_redis_write():
    """Mock redis_write function."""
    with patch("common.providers.universal_memory_provider.redis_write") as mock:
        mock.return_value = {"success": True}
        yield mock


@pytest.fixture
def mock_redis_read():
    """Mock redis_read function."""
    with patch("common.providers.universal_memory_provider.redis_read") as mock:
        yield mock


@pytest.fixture
def mock_redis_get_keys():
    """Mock redis_get_keys function."""
    with patch("common.providers.universal_memory_provider.redis_get_keys") as mock:
        yield mock


class TestWriteMemory:
    """Test writing memories to Redis."""

    def test_write_memory_success(self, memory_provider, mock_redis_write):
        """Test successfully writing a memory."""
        memory_id = memory_provider.write_memory(
            user_id="user123",
            memory_type="conversation",
            content="Test memory content",
            source_role="conversation",
            importance=0.7,
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

        # Verify redis_write was called
        assert mock_redis_write.call_count >= 1  # Main write + indices

    def test_write_memory_with_metadata(self, memory_provider, mock_redis_write):
        """Test writing memory with metadata and tags."""
        metadata = {"key": "value", "number": 42}
        tags = ["tag1", "tag2"]
        related = ["mem-abc", "mem-def"]

        memory_id = memory_provider.write_memory(
            user_id="user123",
            memory_type="event",
            content="Event content",
            source_role="calendar",
            importance=0.8,
            metadata=metadata,
            tags=tags,
            related_memories=related,
        )

        assert memory_id is not None

        # Verify the data passed to redis_write
        call_args = mock_redis_write.call_args_list[0]
        memory_data = call_args[0][1]

        assert memory_data["metadata"] == metadata
        assert memory_data["tags"] == tags
        assert memory_data["related_memories"] == related

    def test_write_memory_calculates_ttl(self, memory_provider, mock_redis_write):
        """Test TTL calculation based on importance."""
        # Low importance
        memory_provider.write_memory(
            user_id="user123",
            memory_type="conversation",
            content="Low importance",
            source_role="conversation",
            importance=0.1,
        )

        low_ttl = mock_redis_write.call_args_list[0][1]["ttl"]

        # High importance
        mock_redis_write.reset_mock()
        memory_provider.write_memory(
            user_id="user123",
            memory_type="conversation",
            content="High importance",
            source_role="conversation",
            importance=0.9,
        )

        high_ttl = mock_redis_write.call_args_list[0][1]["ttl"]

        # High importance should have longer TTL
        assert high_ttl > low_ttl

    def test_write_memory_failure(self, memory_provider, mock_redis_write):
        """Test handling write failure."""
        mock_redis_write.return_value = {"success": False, "error": "Redis error"}

        memory_id = memory_provider.write_memory(
            user_id="user123",
            memory_type="conversation",
            content="Test",
            source_role="conversation",
        )

        assert memory_id is None

    def test_write_memory_exception(self, memory_provider, mock_redis_write):
        """Test handling exceptions during write."""
        mock_redis_write.side_effect = Exception("Unexpected error")

        memory_id = memory_provider.write_memory(
            user_id="user123",
            memory_type="conversation",
            content="Test",
            source_role="conversation",
        )

        assert memory_id is None


class TestSearchMemories:
    """Test searching memories with various filters."""

    def test_search_memories_by_query(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test keyword search across memories."""
        # Setup mock data
        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": ["memory:user123:id1", "memory:user123:id2"],
        }

        memory1 = UniversalMemory(
            id="id1",
            user_id="user123",
            memory_type="conversation",
            content="Meeting about project planning",
            source_role="conversation",
            timestamp=time.time(),
            importance=0.7,
        )

        memory2 = UniversalMemory(
            id="id2",
            user_id="user123",
            memory_type="event",
            content="Lunch with team",
            source_role="calendar",
            timestamp=time.time(),
            importance=0.5,
        )

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1.to_dict()},
            {"success": True, "value": memory2.to_dict()},
        ]

        # Search for "meeting"
        results = memory_provider.search_memories(
            user_id="user123", query="meeting", limit=10
        )

        assert len(results) == 1
        assert results[0].content == "Meeting about project planning"

    def test_search_memories_by_type(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test filtering by memory type."""
        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": ["memory:user123:id1", "memory:user123:id2"],
        }

        memory1 = UniversalMemory(
            id="id1",
            user_id="user123",
            memory_type="conversation",
            content="Chat content",
            source_role="conversation",
            timestamp=time.time(),
        )

        memory2 = UniversalMemory(
            id="id2",
            user_id="user123",
            memory_type="event",
            content="Event content",
            source_role="calendar",
            timestamp=time.time(),
        )

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1.to_dict()},
            {"success": True, "value": memory2.to_dict()},
        ]

        # Search for events only
        results = memory_provider.search_memories(
            user_id="user123", memory_types=["event"], limit=10
        )

        assert len(results) == 1
        assert results[0].memory_type == "event"

    def test_search_memories_by_tags(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test filtering by tags."""
        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": ["memory:user123:id1", "memory:user123:id2"],
        }

        memory1 = UniversalMemory(
            id="id1",
            user_id="user123",
            memory_type="conversation",
            content="Work discussion",
            source_role="conversation",
            timestamp=time.time(),
            tags=["work", "project"],
        )

        memory2 = UniversalMemory(
            id="id2",
            user_id="user123",
            memory_type="conversation",
            content="Personal chat",
            source_role="conversation",
            timestamp=time.time(),
            tags=["personal"],
        )

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1.to_dict()},
            {"success": True, "value": memory2.to_dict()},
        ]

        # Search for work-related memories
        results = memory_provider.search_memories(
            user_id="user123", tags=["work"], limit=10
        )

        assert len(results) == 1
        assert "work" in results[0].tags

    def test_search_memories_by_time_range(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test filtering by timestamp range."""
        now = time.time()
        yesterday = now - 86400
        two_days_ago = now - 172800

        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": ["memory:user123:id1", "memory:user123:id2"],
        }

        memory1 = UniversalMemory(
            id="id1",
            user_id="user123",
            memory_type="conversation",
            content="Recent memory",
            source_role="conversation",
            timestamp=yesterday,
        )

        memory2 = UniversalMemory(
            id="id2",
            user_id="user123",
            memory_type="conversation",
            content="Old memory",
            source_role="conversation",
            timestamp=two_days_ago,
        )

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1.to_dict()},
            {"success": True, "value": memory2.to_dict()},
        ]

        # Search for memories from last day
        results = memory_provider.search_memories(
            user_id="user123", start_time=yesterday - 3600, limit=10
        )

        assert len(results) == 1
        assert results[0].content == "Recent memory"

    def test_search_memories_by_importance(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test filtering by minimum importance."""
        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": ["memory:user123:id1", "memory:user123:id2"],
        }

        memory1 = UniversalMemory(
            id="id1",
            user_id="user123",
            memory_type="conversation",
            content="Important memory",
            source_role="conversation",
            timestamp=time.time(),
            importance=0.8,
        )

        memory2 = UniversalMemory(
            id="id2",
            user_id="user123",
            memory_type="conversation",
            content="Less important",
            source_role="conversation",
            timestamp=time.time(),
            importance=0.3,
        )

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1.to_dict()},
            {"success": True, "value": memory2.to_dict()},
        ]

        # Search for important memories only
        results = memory_provider.search_memories(
            user_id="user123", min_importance=0.5, limit=10
        )

        assert len(results) == 1
        assert results[0].importance >= 0.5

    def test_search_memories_respects_limit(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test that search respects the limit parameter."""
        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": [f"memory:user123:id{i}" for i in range(10)],
        }

        # Create 10 memories
        memories = [
            UniversalMemory(
                id=f"id{i}",
                user_id="user123",
                memory_type="conversation",
                content=f"Memory {i}",
                source_role="conversation",
                timestamp=time.time() - i,  # Different timestamps
                importance=0.5,
            )
            for i in range(10)
        ]

        mock_redis_read.side_effect = [
            {"success": True, "value": m.to_dict()} for m in memories
        ]

        # Search with limit of 3
        results = memory_provider.search_memories(user_id="user123", limit=3)

        assert len(results) == 3

    def test_search_memories_no_results(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test search when no memories match."""
        mock_redis_get_keys.return_value = {"success": True, "keys": []}

        results = memory_provider.search_memories(
            user_id="user123", query="nonexistent"
        )

        assert len(results) == 0

    def test_search_memories_redis_failure(self, memory_provider, mock_redis_get_keys):
        """Test handling Redis failure during search."""
        mock_redis_get_keys.return_value = {"success": False, "error": "Redis error"}

        results = memory_provider.search_memories(user_id="user123")

        assert len(results) == 0


class TestGetRecentMemories:
    """Test getting recent memories."""

    def test_get_recent_memories(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test getting recent memories without filters."""
        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": ["memory:user123:id1", "memory:user123:id2"],
        }

        memory1 = UniversalMemory(
            id="id1",
            user_id="user123",
            memory_type="conversation",
            content="Recent chat",
            source_role="conversation",
            timestamp=time.time(),
        )

        memory2 = UniversalMemory(
            id="id2",
            user_id="user123",
            memory_type="event",
            content="Recent event",
            source_role="calendar",
            timestamp=time.time() - 100,
        )

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1.to_dict()},
            {"success": True, "value": memory2.to_dict()},
        ]

        results = memory_provider.get_recent_memories(user_id="user123", limit=5)

        assert len(results) == 2

    def test_get_recent_memories_with_type_filter(
        self, memory_provider, mock_redis_get_keys, mock_redis_read
    ):
        """Test getting recent memories filtered by type."""
        mock_redis_get_keys.return_value = {
            "success": True,
            "keys": ["memory:user123:id1", "memory:user123:id2"],
        }

        memory1 = UniversalMemory(
            id="id1",
            user_id="user123",
            memory_type="conversation",
            content="Chat",
            source_role="conversation",
            timestamp=time.time(),
        )

        memory2 = UniversalMemory(
            id="id2",
            user_id="user123",
            memory_type="event",
            content="Event",
            source_role="calendar",
            timestamp=time.time(),
        )

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1.to_dict()},
            {"success": True, "value": memory2.to_dict()},
        ]

        results = memory_provider.get_recent_memories(
            user_id="user123", memory_types=["conversation"], limit=5
        )

        assert len(results) == 1
        assert results[0].memory_type == "conversation"


class TestLinkMemories:
    """Test linking related memories."""

    def test_link_memories_success(
        self, memory_provider, mock_redis_read, mock_redis_write
    ):
        """Test successfully linking two memories."""
        memory1_data = {
            "id": "id1",
            "user_id": "user123",
            "memory_type": "conversation",
            "content": "Discussed meeting",
            "source_role": "conversation",
            "timestamp": time.time(),
            "importance": 0.5,
            "metadata": {},
            "tags": [],
            "related_memories": [],
        }

        memory2_data = {
            "id": "id2",
            "user_id": "user123",
            "memory_type": "event",
            "content": "Meeting scheduled",
            "source_role": "calendar",
            "timestamp": time.time(),
            "importance": 0.7,
            "metadata": {},
            "tags": [],
            "related_memories": [],
        }

        mock_redis_read.side_effect = [
            {"success": True, "value": memory1_data},
            {"success": True, "value": memory2_data},
        ]

        result = memory_provider.link_memories("id1", "id2", "user123")

        assert result is True
        assert mock_redis_write.call_count == 2

        # Verify cross-references were added
        call_args = mock_redis_write.call_args_list
        updated_mem1 = call_args[0][0][1]
        updated_mem2 = call_args[1][0][1]

        assert "id2" in updated_mem1["related_memories"]
        assert "id1" in updated_mem2["related_memories"]

    def test_link_memories_read_failure(
        self, memory_provider, mock_redis_read, mock_redis_write
    ):
        """Test handling read failure during linking."""
        mock_redis_read.return_value = {"success": False, "error": "Not found"}

        result = memory_provider.link_memories("id1", "id2", "user123")

        assert result is False
        assert mock_redis_write.call_count == 0

    def test_link_memories_exception(
        self, memory_provider, mock_redis_read, mock_redis_write
    ):
        """Test handling exceptions during linking."""
        mock_redis_read.side_effect = Exception("Unexpected error")

        result = memory_provider.link_memories("id1", "id2", "user123")

        assert result is False


class TestTTLCalculation:
    """Test TTL calculation logic."""

    def test_calculate_ttl_low_importance(self, memory_provider):
        """Test TTL for low importance memory."""
        ttl = memory_provider._calculate_ttl(0.0)

        # Should be 30 days
        expected = 30 * 24 * 60 * 60
        assert ttl == expected

    def test_calculate_ttl_high_importance(self, memory_provider):
        """Test TTL for high importance memory."""
        ttl = memory_provider._calculate_ttl(1.0)

        # Should be 90 days
        expected = 90 * 24 * 60 * 60
        assert ttl == expected

    def test_calculate_ttl_medium_importance(self, memory_provider):
        """Test TTL for medium importance memory."""
        ttl = memory_provider._calculate_ttl(0.5)

        # Should be between 30 and 90 days
        min_ttl = 30 * 24 * 60 * 60
        max_ttl = 90 * 24 * 60 * 60
        assert min_ttl < ttl < max_ttl
