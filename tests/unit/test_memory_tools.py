"""Tests for memory tools."""

from unittest.mock import MagicMock, patch

import pytest

from common.providers.universal_memory_provider import UniversalMemory


@pytest.fixture
def mock_memory_provider():
    """Mock UniversalMemoryProvider."""
    with patch(
        "roles.shared_tools.memory_tools._get_memory_provider"
    ) as mock_get_provider:
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def mock_get_user_id():
    """Mock _get_current_user_id function."""
    with patch("roles.shared_tools.memory_tools._get_current_user_id") as mock:
        mock.return_value = "test_user"
        yield mock


class TestSearchMemoryTool:
    """Test search_memory tool."""

    def test_search_memory_basic(self, mock_memory_provider, mock_get_user_id):
        """Test basic memory search."""
        from roles.shared_tools.memory_tools import search_memory

        # Setup mock
        mock_memory = UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="conversation",
            content="Test memory content",
            source_role="conversation",
            timestamp=1234567890.0,
            importance=0.7,
            tags=["test"],
        )
        mock_memory_provider.search_memories.return_value = [mock_memory]

        # Execute
        result = search_memory(query="test")

        # Verify
        assert result["success"] is True
        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == "Test memory content"
        assert result["memories"][0]["type"] == "conversation"
        assert result["count"] == 1

        # Verify provider was called correctly
        mock_memory_provider.search_memories.assert_called_once_with(
            user_id="test_user", query="test", memory_types=None, tags=None, limit=10
        )

    def test_search_memory_with_filters(self, mock_memory_provider, mock_get_user_id):
        """Test memory search with type and tag filters."""
        from roles.shared_tools.memory_tools import search_memory

        mock_memory = UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="event",
            content="Event content",
            source_role="calendar",
            timestamp=1234567890.0,
            tags=["meeting"],
        )
        mock_memory_provider.search_memories.return_value = [mock_memory]

        # Execute with filters
        result = search_memory(
            query="meeting", memory_types=["event"], tags=["meeting"], limit=5
        )

        # Verify
        assert result["success"] is True
        assert len(result["memories"]) == 1
        assert result["memories"][0]["type"] == "event"

        # Verify filters were passed
        mock_memory_provider.search_memories.assert_called_once_with(
            user_id="test_user",
            query="meeting",
            memory_types=["event"],
            tags=["meeting"],
            limit=5,
        )

    def test_search_memory_no_results(self, mock_memory_provider, mock_get_user_id):
        """Test search with no results."""
        from roles.shared_tools.memory_tools import search_memory

        mock_memory_provider.search_memories.return_value = []

        result = search_memory(query="nonexistent")

        assert result["success"] is True
        assert len(result["memories"]) == 0
        assert result["count"] == 0

    def test_search_memory_with_metadata(self, mock_memory_provider, mock_get_user_id):
        """Test that search returns metadata."""
        from roles.shared_tools.memory_tools import search_memory

        mock_memory = UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="plan",
            content="Plan content",
            source_role="planning",
            timestamp=1234567890.0,
            importance=0.8,
            metadata={"workflow_id": "wf123"},
            tags=["project"],
        )
        mock_memory_provider.search_memories.return_value = [mock_memory]

        result = search_memory(query="plan")

        assert result["success"] is True
        memory = result["memories"][0]
        assert memory["content"] == "Plan content"
        assert memory["type"] == "plan"
        assert memory["source"] == "planning"
        assert memory["importance"] == 0.8
        assert memory["tags"] == ["project"]
        assert memory["metadata"] == {"workflow_id": "wf123"}

    def test_search_memory_error_handling(self, mock_memory_provider, mock_get_user_id):
        """Test error handling in search."""
        from roles.shared_tools.memory_tools import search_memory

        mock_memory_provider.search_memories.side_effect = Exception("Redis error")

        result = search_memory(query="test")

        assert result["success"] is False
        assert "error" in result
        assert "Redis error" in result["error"]


class TestGetRecentMemoriesTool:
    """Test get_recent_memories tool."""

    def test_get_recent_memories_basic(self, mock_memory_provider, mock_get_user_id):
        """Test getting recent memories."""
        from roles.shared_tools.memory_tools import get_recent_memories

        mock_memories = [
            UniversalMemory(
                id=f"mem{i}",
                user_id="test_user",
                memory_type="conversation",
                content=f"Memory {i}",
                source_role="conversation",
                timestamp=1234567890.0 - i,
            )
            for i in range(3)
        ]
        mock_memory_provider.get_recent_memories.return_value = mock_memories

        result = get_recent_memories()

        assert result["success"] is True
        assert len(result["memories"]) == 3
        assert result["memories"][0]["content"] == "Memory 0"

        mock_memory_provider.get_recent_memories.assert_called_once_with(
            user_id="test_user", memory_types=None, limit=10
        )

    def test_get_recent_memories_with_type_filter(
        self, mock_memory_provider, mock_get_user_id
    ):
        """Test getting recent memories filtered by type."""
        from roles.shared_tools.memory_tools import get_recent_memories

        mock_memory = UniversalMemory(
            id="mem1",
            user_id="test_user",
            memory_type="event",
            content="Event",
            source_role="calendar",
            timestamp=1234567890.0,
        )
        mock_memory_provider.get_recent_memories.return_value = [mock_memory]

        result = get_recent_memories(memory_types=["event"], limit=5)

        assert result["success"] is True
        assert len(result["memories"]) == 1
        assert result["memories"][0]["type"] == "event"

        mock_memory_provider.get_recent_memories.assert_called_once_with(
            user_id="test_user", memory_types=["event"], limit=5
        )

    def test_get_recent_memories_empty(self, mock_memory_provider, mock_get_user_id):
        """Test getting recent memories when none exist."""
        from roles.shared_tools.memory_tools import get_recent_memories

        mock_memory_provider.get_recent_memories.return_value = []

        result = get_recent_memories()

        assert result["success"] is True
        assert len(result["memories"]) == 0

    def test_get_recent_memories_error_handling(
        self, mock_memory_provider, mock_get_user_id
    ):
        """Test error handling."""
        from roles.shared_tools.memory_tools import get_recent_memories

        mock_memory_provider.get_recent_memories.side_effect = Exception("Error")

        result = get_recent_memories()

        assert result["success"] is False
        assert "error" in result


class TestGetCurrentUserId:
    """Test _get_current_user_id helper."""

    def test_get_current_user_id_from_context(self):
        """Test getting user ID from context."""
        from roles.shared_tools.memory_tools import _get_current_user_id

        # Mock context in thread local storage
        with patch("roles.shared_tools.memory_tools._context_storage") as mock_storage:
            mock_context = MagicMock()
            mock_context.user_id = "user123"
            mock_storage.context = mock_context

            user_id = _get_current_user_id()

            assert user_id == "user123"

    def test_get_current_user_id_fallback(self):
        """Test fallback when no context available."""
        from roles.shared_tools.memory_tools import _get_current_user_id

        with patch("roles.shared_tools.memory_tools._context_storage") as mock_storage:
            mock_storage.context = None

            user_id = _get_current_user_id()

            assert user_id == "unknown"

    def test_get_current_user_id_no_user_id_attribute(self):
        """Test fallback when context has no user_id."""
        from roles.shared_tools.memory_tools import _get_current_user_id

        with patch("roles.shared_tools.memory_tools._context_storage") as mock_storage:
            mock_context = MagicMock(spec=[])  # No user_id attribute
            mock_storage.context = mock_context

            user_id = _get_current_user_id()

            assert user_id == "unknown"
