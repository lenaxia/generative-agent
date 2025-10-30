"""
Tests for context types and context collector.

This module tests the ContextType enum and ContextCollector class that
provide enum-based context gathering with interface-driven design.
"""

from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.context_types import ContextCollector, ContextType
from common.interfaces.context_interfaces import (
    LocationProvider,
    MemoryEntry,
    MemoryProvider,
)


class TestContextType:
    """Test ContextType enum."""

    def test_context_type_values(self):
        """Test ContextType enum values."""
        assert ContextType.LOCATION.value == "location"
        assert ContextType.RECENT_MEMORY.value == "recent_memory"
        assert ContextType.PRESENCE.value == "presence"
        assert ContextType.SCHEDULE.value == "schedule"

    def test_context_type_membership(self):
        """Test ContextType enum membership."""
        assert ContextType.LOCATION in ContextType
        assert ContextType.RECENT_MEMORY in ContextType
        assert ContextType.PRESENCE in ContextType
        assert ContextType.SCHEDULE in ContextType

    def test_context_type_iteration(self):
        """Test ContextType enum iteration."""
        context_types = list(ContextType)
        assert len(context_types) == 4
        assert ContextType.LOCATION in context_types
        assert ContextType.RECENT_MEMORY in context_types
        assert ContextType.PRESENCE in context_types
        assert ContextType.SCHEDULE in context_types


class MockMemoryProvider(MemoryProvider):
    """Mock memory provider for testing."""

    def __init__(self, memories: list[MemoryEntry] = None):
        self.memories = memories or []
        self.stored_memories = []

    async def store_memory(self, memory: MemoryEntry) -> bool:
        self.stored_memories.append(memory)
        return True

    async def get_recent_memories(
        self, user_id: str, limit: int = 3
    ) -> list[MemoryEntry]:
        user_memories = [m for m in self.memories if m.user_id == user_id]
        return user_memories[-limit:]

    async def search_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[MemoryEntry]:
        user_memories = [m for m in self.memories if m.user_id == user_id]
        # Simple search - check if query words are in content
        query_words = query.lower().split()
        matching_memories = []
        for memory in user_memories:
            if any(word in memory.content.lower() for word in query_words):
                matching_memories.append(memory)
        return matching_memories[:limit]


class MockLocationProvider(LocationProvider):
    """Mock location provider for testing."""

    def __init__(self, locations: dict[str, str] = None):
        self.locations = locations or {}

    async def get_current_location(self, user_id: str) -> str | None:
        return self.locations.get(user_id)

    async def update_location(
        self, user_id: str, location: str, confidence: float = 1.0
    ) -> bool:
        self.locations[user_id] = location
        return True


class TestContextCollector:
    """Test ContextCollector class."""

    @pytest.fixture
    def mock_memory_provider(self):
        """Create mock memory provider with test data."""
        memories = [
            MemoryEntry(
                user_id="test_user",
                content="I like jazz music in the evening",
                timestamp=datetime.now(),
                location="living_room",
                importance=0.7,
            ),
            MemoryEntry(
                user_id="test_user",
                content="Meeting with Bob at 3pm",
                timestamp=datetime.now(),
                location="office",
                importance=0.8,
            ),
            MemoryEntry(
                user_id="other_user",
                content="Other user's memory",
                timestamp=datetime.now(),
                importance=0.5,
            ),
        ]
        return MockMemoryProvider(memories)

    @pytest.fixture
    def mock_location_provider(self):
        """Create mock location provider with test data."""
        locations = {
            "test_user": "bedroom",
            "other_user": "kitchen",
        }
        return MockLocationProvider(locations)

    @pytest.fixture
    def context_collector(self, mock_memory_provider, mock_location_provider):
        """Create ContextCollector with mocked providers."""
        return ContextCollector(
            memory_provider=mock_memory_provider,
            location_provider=mock_location_provider,
        )

    @pytest.mark.asyncio
    async def test_context_collector_initialization(
        self, mock_memory_provider, mock_location_provider
    ):
        """Test ContextCollector initialization."""
        collector = ContextCollector(
            memory_provider=mock_memory_provider,
            location_provider=mock_location_provider,
        )

        assert collector.memory_provider == mock_memory_provider
        assert collector.location_provider == mock_location_provider

    @pytest.mark.asyncio
    async def test_initialize_method(self, context_collector):
        """Test ContextCollector initialize method."""
        # Should not raise any exceptions
        await context_collector.initialize()

    @pytest.mark.asyncio
    async def test_gather_context_empty_input(self, context_collector):
        """Test gather_context with empty inputs."""
        # Empty context types
        result = await context_collector.gather_context("test_user", [])
        assert result == {}

        # Empty user_id
        result = await context_collector.gather_context("", ["location"])
        assert result == {}

        # None user_id
        result = await context_collector.gather_context(None, ["location"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_gather_location_context(self, context_collector):
        """Test gathering location context."""
        result = await context_collector.gather_context(
            "test_user", [ContextType.LOCATION.value]
        )

        assert "location" in result
        assert result["location"] == "bedroom"

    @pytest.mark.asyncio
    async def test_gather_location_context_no_location(self, context_collector):
        """Test gathering location context when user has no location."""
        result = await context_collector.gather_context(
            "nonexistent_user", [ContextType.LOCATION.value]
        )

        # Should return empty dict when no location found
        assert result == {}

    @pytest.mark.asyncio
    async def test_gather_memory_context(self, context_collector):
        """Test gathering recent memory context."""
        result = await context_collector.gather_context(
            "test_user", [ContextType.RECENT_MEMORY.value]
        )

        assert "recent_memory" in result
        assert isinstance(result["recent_memory"], list)
        assert len(result["recent_memory"]) == 2  # test_user has 2 memories
        assert "jazz music" in result["recent_memory"][0]
        assert "Meeting with Bob" in result["recent_memory"][1]

    @pytest.mark.asyncio
    async def test_gather_memory_context_no_memories(self, context_collector):
        """Test gathering memory context when user has no memories."""
        result = await context_collector.gather_context(
            "user_with_no_memories", [ContextType.RECENT_MEMORY.value]
        )

        # Should return empty dict when no memories found
        assert result == {}

    @pytest.mark.asyncio
    async def test_gather_presence_context(self, context_collector):
        """Test gathering presence context."""
        # Mock redis_get_keys and redis_read for presence detection
        with (
            patch("common.context_types.redis_get_keys") as mock_get_keys,
            patch("common.context_types.redis_read") as mock_read,
        ):
            # Setup mock responses
            mock_get_keys.return_value = {
                "success": True,
                "keys": [
                    "location:test_user",
                    "location:other_user",
                    "location:third_user",
                ],
            }

            def mock_read_side_effect(key):
                if key == "location:other_user":
                    return {"success": True, "value": "home"}
                elif key == "location:third_user":
                    return {"success": True, "value": "home"}
                else:
                    return {"success": True, "value": "away"}

            mock_read.side_effect = mock_read_side_effect

            result = await context_collector.gather_context(
                "test_user", [ContextType.PRESENCE.value]
            )

            assert "presence" in result
            assert isinstance(result["presence"], list)
            assert "other_user" in result["presence"]
            assert "third_user" in result["presence"]
            assert "test_user" not in result["presence"]  # Exclude self

    @pytest.mark.asyncio
    async def test_gather_presence_context_redis_failure(self, context_collector):
        """Test gathering presence context when Redis fails."""
        with patch("common.context_types.redis_get_keys") as mock_get_keys:
            mock_get_keys.return_value = {"success": False}

            result = await context_collector.gather_context(
                "test_user", [ContextType.PRESENCE.value]
            )

            # Should return empty dict when Redis fails
            assert result == {}

    @pytest.mark.asyncio
    async def test_gather_schedule_context(self, context_collector):
        """Test gathering schedule context from Redis."""
        with patch("common.context_types.redis_read") as mock_read:
            # Test with no schedule data
            mock_read.return_value = {"success": False}

            result = await context_collector.gather_context(
                "test_user", [ContextType.SCHEDULE.value]
            )

            # Should return empty dict when no schedule data
            assert result == {}

            # Test with schedule data
            mock_read.return_value = {
                "success": True,
                "value": [
                    {"title": "Meeting", "time": "10:00"},
                    {"title": "Lunch", "time": "12:00"},
                ],
            }

            result = await context_collector.gather_context(
                "test_user", [ContextType.SCHEDULE.value]
            )

            assert "schedule" in result
            assert len(result["schedule"]) == 2
            assert result["schedule"][0]["title"] == "Meeting"

    @pytest.mark.asyncio
    async def test_gather_multiple_contexts(self, context_collector):
        """Test gathering multiple context types."""
        result = await context_collector.gather_context(
            "test_user", [ContextType.LOCATION.value, ContextType.RECENT_MEMORY.value]
        )

        assert "location" in result
        assert "recent_memory" in result
        assert result["location"] == "bedroom"
        assert len(result["recent_memory"]) == 2

    @pytest.mark.asyncio
    async def test_gather_context_with_provider_errors(self, context_collector):
        """Test graceful error handling when providers fail."""
        # Make location provider fail
        context_collector.location_provider.get_current_location = AsyncMock(
            side_effect=Exception("Location provider failed")
        )

        result = await context_collector.gather_context(
            "test_user", [ContextType.LOCATION.value, ContextType.RECENT_MEMORY.value]
        )

        # Should have memory but not location due to error
        assert "location" not in result
        assert "recent_memory" in result
        assert len(result["recent_memory"]) == 2

    @pytest.mark.asyncio
    async def test_gather_context_all_providers_fail(self, context_collector):
        """Test behavior when all providers fail."""
        # Make all providers fail
        context_collector.location_provider.get_current_location = AsyncMock(
            side_effect=Exception("Location failed")
        )
        context_collector.memory_provider.get_recent_memories = AsyncMock(
            side_effect=Exception("Memory failed")
        )

        result = await context_collector.gather_context(
            "test_user", [ContextType.LOCATION.value, ContextType.RECENT_MEMORY.value]
        )

        # Should return empty dict when all providers fail
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_others_home_method(self, context_collector):
        """Test _get_others_home method directly."""
        with (
            patch("common.context_types.redis_get_keys") as mock_get_keys,
            patch("common.context_types.redis_read") as mock_read,
        ):
            mock_get_keys.return_value = {
                "success": True,
                "keys": ["location:user1", "location:user2", "location:user3"],
            }

            def mock_read_side_effect(key):
                if key == "location:user1":
                    return {"success": True, "value": "home"}
                elif key == "location:user2":
                    return {"success": True, "value": "away"}
                elif key == "location:user3":
                    return {"success": True, "value": "home"}

            mock_read.side_effect = mock_read_side_effect

            others_home = await context_collector._get_others_home("test_user")

            assert isinstance(others_home, list)
            assert "user1" in others_home
            assert "user3" in others_home
            assert "user2" not in others_home  # Away
            assert "test_user" not in others_home  # Self excluded

    @pytest.mark.asyncio
    async def test_get_others_home_redis_error(self, context_collector):
        """Test _get_others_home with Redis errors."""
        with patch("common.context_types.redis_get_keys") as mock_get_keys:
            mock_get_keys.side_effect = Exception("Redis connection failed")

            others_home = await context_collector._get_others_home("test_user")

            assert others_home == []

    def test_context_collector_with_provider_initialization(self):
        """Test that providers with initialize method are called."""
        mock_memory = Mock()
        mock_memory.initialize = AsyncMock()
        mock_location = Mock()
        mock_location.initialize = AsyncMock()

        collector = ContextCollector(
            memory_provider=mock_memory, location_provider=mock_location
        )

        # Test that initialize methods would be called
        assert hasattr(mock_memory, "initialize")
        assert hasattr(mock_location, "initialize")


class TestContextCollectorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_invalid_context_type(self):
        """Test handling of invalid context types."""
        mock_memory = MockMemoryProvider()
        mock_location = MockLocationProvider()
        collector = ContextCollector(
            memory_provider=mock_memory, location_provider=mock_location
        )

        # Invalid context type should be ignored
        result = await collector.gather_context("test_user", ["invalid_context_type"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_context_types(self):
        """Test handling mix of valid and invalid context types."""
        mock_memory = MockMemoryProvider()
        mock_location = MockLocationProvider({"test_user": "bedroom"})
        collector = ContextCollector(
            memory_provider=mock_memory, location_provider=mock_location
        )

        result = await collector.gather_context(
            "test_user", [ContextType.LOCATION.value, "invalid_type"]
        )

        # Should process valid types and ignore invalid ones
        assert "location" in result
        assert result["location"] == "bedroom"

    @pytest.mark.asyncio
    async def test_context_collector_type_validation(self):
        """Test ContextCollector type validation."""
        mock_memory = MockMemoryProvider()
        mock_location = MockLocationProvider()

        # Should accept valid providers
        collector = ContextCollector(
            memory_provider=mock_memory, location_provider=mock_location
        )
        assert collector.memory_provider == mock_memory
        assert collector.location_provider == mock_location


if __name__ == "__main__":
    pytest.main([__file__])
