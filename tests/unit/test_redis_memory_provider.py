"""
Tests for Redis memory provider implementation.

This module tests the RedisMemoryProvider class that implements the MemoryProvider
interface using Redis as the backend storage system.
"""

from datetime import datetime
from typing import List
from unittest.mock import Mock, patch

import pytest

from common.interfaces.context_interfaces import MemoryEntry
from common.providers.redis_memory_provider import RedisMemoryProvider


class TestRedisMemoryProvider:
    """Test RedisMemoryProvider implementation."""

    @pytest.fixture
    def memory_provider(self):
        """Create RedisMemoryProvider instance."""
        return RedisMemoryProvider()

    @pytest.fixture
    def sample_memory_entry(self):
        """Create sample memory entry for testing."""
        return MemoryEntry(
            user_id="test_user",
            content="I like jazz music in the evening",
            timestamp=datetime(2023, 10, 15, 20, 30, 0),
            location="living_room",
            importance=0.7,
            metadata={"source": "conversation", "topic": "music"},
        )

    def test_memory_provider_initialization(self, memory_provider):
        """Test RedisMemoryProvider initialization."""
        assert memory_provider.key_prefix == "memory"
        assert isinstance(memory_provider, RedisMemoryProvider)

    @pytest.mark.asyncio
    async def test_store_memory_success(self, memory_provider, sample_memory_entry):
        """Test successful memory storage."""
        with patch("common.providers.redis_memory_provider.redis_write") as mock_write:
            mock_write.return_value = {"success": True}

            result = await memory_provider.store_memory(sample_memory_entry)

            assert result is True
            mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_memory_failure(self, memory_provider, sample_memory_entry):
        """Test memory storage failure."""
        with patch("common.providers.redis_memory_provider.redis_write") as mock_write:
            mock_write.return_value = {"success": False}

            result = await memory_provider.store_memory(sample_memory_entry)

            assert result is False

    @pytest.mark.asyncio
    async def test_get_recent_memories_success(self, memory_provider):
        """Test successful retrieval of recent memories."""
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
                    "content": "Test memory",
                    "timestamp": "2023-10-15T20:00:00",
                    "location": "office",
                    "importance": 0.5,
                    "metadata": {},
                },
            }

            memories = await memory_provider.get_recent_memories("test_user", limit=3)

            assert len(memories) == 1
            assert isinstance(memories[0], MemoryEntry)
            assert memories[0].content == "Test memory"

    @pytest.mark.asyncio
    async def test_search_memories_success(self, memory_provider):
        """Test successful memory search."""
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
                    "content": "I love jazz music",
                    "timestamp": "2023-10-15T20:00:00",
                    "location": "living_room",
                    "importance": 0.7,
                    "metadata": {},
                },
            }

            memories = await memory_provider.search_memories("test_user", "music")

            assert len(memories) == 1
            assert "jazz music" in memories[0].content


if __name__ == "__main__":
    pytest.main([__file__])
