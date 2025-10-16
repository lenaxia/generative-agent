"""
Tests for context interfaces and data structures.

This module tests the abstract interfaces and data classes that define
the contract for context providers in the Universal Agent System.
"""

from datetime import datetime
from typing import List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from common.interfaces.context_interfaces import (
    ContextData,
    ContextProvider,
    EnvironmentProvider,
    LocationData,
    LocationProvider,
    MemoryEntry,
    MemoryProvider,
)


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test basic MemoryEntry creation."""
        timestamp = datetime.now()
        entry = MemoryEntry(
            user_id="test_user",
            content="Test memory content",
            timestamp=timestamp,
            location="living_room",
            importance=0.8,
            metadata={"source": "test"},
        )

        assert entry.user_id == "test_user"
        assert entry.content == "Test memory content"
        assert entry.timestamp == timestamp
        assert entry.location == "living_room"
        assert entry.importance == 0.8
        assert entry.metadata == {"source": "test"}

    def test_memory_entry_minimal(self):
        """Test MemoryEntry with minimal required fields."""
        timestamp = datetime.now()
        entry = MemoryEntry(
            user_id="test_user", content="Test content", timestamp=timestamp
        )

        assert entry.user_id == "test_user"
        assert entry.content == "Test content"
        assert entry.timestamp == timestamp
        assert entry.location is None
        assert entry.importance == 0.5  # Default value
        assert entry.metadata is None

    def test_memory_entry_validation(self):
        """Test MemoryEntry field validation."""
        timestamp = datetime.now()

        # Valid entry
        entry = MemoryEntry(
            user_id="test_user",
            content="Valid content",
            timestamp=timestamp,
            importance=0.7,
        )
        assert entry.importance == 0.7

        # Test importance bounds (should be handled by validation if implemented)
        entry_low = MemoryEntry(
            user_id="test_user",
            content="Low importance",
            timestamp=timestamp,
            importance=0.0,
        )
        assert entry_low.importance == 0.0

        entry_high = MemoryEntry(
            user_id="test_user",
            content="High importance",
            timestamp=timestamp,
            importance=1.0,
        )
        assert entry_high.importance == 1.0


class TestLocationData:
    """Test LocationData dataclass."""

    def test_location_data_creation(self):
        """Test basic LocationData creation."""
        timestamp = datetime.now()
        location_data = LocationData(
            user_id="test_user",
            current_location="bedroom",
            previous_location="kitchen",
            timestamp=timestamp,
            confidence=0.95,
        )

        assert location_data.user_id == "test_user"
        assert location_data.current_location == "bedroom"
        assert location_data.previous_location == "kitchen"
        assert location_data.timestamp == timestamp
        assert location_data.confidence == 0.95

    def test_location_data_minimal(self):
        """Test LocationData with minimal required fields."""
        timestamp = datetime.now()
        location_data = LocationData(
            user_id="test_user", current_location="living_room", timestamp=timestamp
        )

        assert location_data.user_id == "test_user"
        assert location_data.current_location == "living_room"
        assert location_data.previous_location is None
        assert location_data.timestamp == timestamp
        assert location_data.confidence == 1.0  # Default value


class TestContextData:
    """Test ContextData dataclass."""

    def test_context_data_creation(self):
        """Test basic ContextData creation."""
        context_data = ContextData(
            user_id="test_user",
            context_type="location",
            data={"current_location": "bedroom"},
            metadata={"source": "mqtt", "confidence": 0.9},
        )

        assert context_data.user_id == "test_user"
        assert context_data.context_type == "location"
        assert context_data.data == {"current_location": "bedroom"}
        assert context_data.metadata == {"source": "mqtt", "confidence": 0.9}

    def test_context_data_minimal(self):
        """Test ContextData with minimal required fields."""
        context_data = ContextData(
            user_id="test_user", context_type="memory", data={"recent_memories": []}
        )

        assert context_data.user_id == "test_user"
        assert context_data.context_type == "memory"
        assert context_data.data == {"recent_memories": []}
        assert context_data.metadata is None


class TestMemoryProviderInterface:
    """Test MemoryProvider abstract interface."""

    def test_memory_provider_is_abstract(self):
        """Test that MemoryProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryProvider()

    def test_memory_provider_interface_methods(self):
        """Test that MemoryProvider defines required abstract methods."""
        # Check that abstract methods exist
        assert hasattr(MemoryProvider, "store_memory")
        assert hasattr(MemoryProvider, "get_recent_memories")
        assert hasattr(MemoryProvider, "search_memories")

        # Verify methods are abstract
        assert getattr(MemoryProvider.store_memory, "__isabstractmethod__", False)
        assert getattr(
            MemoryProvider.get_recent_memories, "__isabstractmethod__", False
        )
        assert getattr(MemoryProvider.search_memories, "__isabstractmethod__", False)


class TestLocationProviderInterface:
    """Test LocationProvider abstract interface."""

    def test_location_provider_is_abstract(self):
        """Test that LocationProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LocationProvider()

    def test_location_provider_interface_methods(self):
        """Test that LocationProvider defines required abstract methods."""
        # Check that abstract methods exist
        assert hasattr(LocationProvider, "get_current_location")
        assert hasattr(LocationProvider, "update_location")

        # Verify methods are abstract
        assert getattr(
            LocationProvider.get_current_location, "__isabstractmethod__", False
        )
        assert getattr(LocationProvider.update_location, "__isabstractmethod__", False)


class TestContextProviderInterface:
    """Test ContextProvider abstract interface."""

    def test_context_provider_is_abstract(self):
        """Test that ContextProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ContextProvider()

    def test_context_provider_interface_methods(self):
        """Test that ContextProvider defines required abstract methods."""
        # Check that abstract methods exist
        assert hasattr(ContextProvider, "get_context")

        # Verify method is abstract
        assert getattr(ContextProvider.get_context, "__isabstractmethod__", False)


class TestEnvironmentProviderInterface:
    """Test EnvironmentProvider abstract interface."""

    def test_environment_provider_is_abstract(self):
        """Test that EnvironmentProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EnvironmentProvider()

    def test_environment_provider_interface_methods(self):
        """Test that EnvironmentProvider defines required abstract methods."""
        # Check that abstract methods exist
        assert hasattr(EnvironmentProvider, "get_environment_data")

        # Verify method is abstract
        assert getattr(
            EnvironmentProvider.get_environment_data, "__isabstractmethod__", False
        )


class MockMemoryProvider(MemoryProvider):
    """Mock implementation of MemoryProvider for testing."""

    async def store_memory(self, memory: MemoryEntry) -> bool:
        return True

    async def get_recent_memories(
        self, user_id: str, limit: int = 3
    ) -> list[MemoryEntry]:
        return []

    async def search_memories(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[MemoryEntry]:
        return []


class MockLocationProvider(LocationProvider):
    """Mock implementation of LocationProvider for testing."""

    async def get_current_location(self, user_id: str) -> Optional[str]:
        return "test_location"

    async def update_location(
        self, user_id: str, location: str, confidence: float = 1.0
    ) -> bool:
        return True


class TestProviderImplementations:
    """Test that concrete implementations work correctly."""

    @pytest.mark.asyncio
    async def test_mock_memory_provider(self):
        """Test mock memory provider implementation."""
        provider = MockMemoryProvider()

        # Test store_memory
        memory = MemoryEntry(
            user_id="test_user", content="Test memory", timestamp=datetime.now()
        )
        result = await provider.store_memory(memory)
        assert result is True

        # Test get_recent_memories
        memories = await provider.get_recent_memories("test_user", limit=5)
        assert isinstance(memories, list)
        assert len(memories) == 0  # Mock returns empty list

        # Test search_memories
        search_results = await provider.search_memories("test_user", "test query")
        assert isinstance(search_results, list)
        assert len(search_results) == 0  # Mock returns empty list

    @pytest.mark.asyncio
    async def test_mock_location_provider(self):
        """Test mock location provider implementation."""
        provider = MockLocationProvider()

        # Test get_current_location
        location = await provider.get_current_location("test_user")
        assert location == "test_location"

        # Test update_location
        result = await provider.update_location(
            "test_user", "new_location", confidence=0.9
        )
        assert result is True


class TestInterfaceContracts:
    """Test interface contracts and type hints."""

    def test_memory_entry_type_hints(self):
        """Test MemoryEntry type annotations."""
        # This test ensures type hints are properly defined
        annotations = MemoryEntry.__annotations__

        assert "user_id" in annotations
        assert "content" in annotations
        assert "timestamp" in annotations
        assert "location" in annotations
        assert "importance" in annotations
        assert "metadata" in annotations

    def test_location_data_type_hints(self):
        """Test LocationData type annotations."""
        annotations = LocationData.__annotations__

        assert "user_id" in annotations
        assert "current_location" in annotations
        assert "previous_location" in annotations
        assert "timestamp" in annotations
        assert "confidence" in annotations

    def test_context_data_type_hints(self):
        """Test ContextData type annotations."""
        annotations = ContextData.__annotations__

        assert "user_id" in annotations
        assert "context_type" in annotations
        assert "data" in annotations
        assert "metadata" in annotations


if __name__ == "__main__":
    pytest.main([__file__])
