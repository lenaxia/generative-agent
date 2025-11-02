"""Tests for planning role unified memory integration."""

from unittest.mock import MagicMock, patch

import pytest

from common.intents import MemoryWriteIntent
from common.providers.universal_memory_provider import UniversalMemory


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
def mock_context():
    """Mock context object."""
    context = MagicMock()
    context.user_id = "test_user"
    context.channel_id = "test_channel"
    return context


class TestPlanningMemoryIntegration:
    """Test planning role integration with unified memory."""

    def test_planning_loads_plan_memories(self, mock_memory_provider):
        """Test planning pre-processing loads past plan memories."""
        from roles.core_planning import load_planning_context

        # Setup mock
        mock_memories = [
            UniversalMemory(
                id=f"mem{i}",
                user_id="test_user",
                memory_type="plan",
                content=f"Plan result {i}",
                source_role="planning",
                timestamp=1234567890.0 - i,
                tags=["project"],
            )
            for i in range(5)
        ]
        mock_memory_provider.get_recent_memories.return_value = mock_memories

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        # Execute
        result = load_planning_context("test instruction", mock_context, {})

        # Verify Tier 1 memories were loaded
        assert "tier1_memories" in result
        assert len(result["tier1_memories"]) == 5

        # Verify provider was called correctly
        mock_memory_provider.get_recent_memories.assert_called_once_with(
            user_id="test_user", memory_types=["plan", "conversation"], limit=5
        )

    def test_planning_has_search_memory_tool(self):
        """Test planning role has search_memory tool."""
        from roles.core_planning import register_role

        registration = register_role()

        # Check that memory_tools is in shared tools
        assert "tools" in registration["config"]
        assert "shared" in registration["config"]["tools"]
        assert "memory_tools" in registration["config"]["tools"]["shared"]

    def test_planning_emits_memory_write_intent(self, mock_context):
        """Test planning post-processing emits MemoryWriteIntent."""
        from roles.core_planning import save_planning_result

        llm_result = "Created plan with 3 tasks"

        # Execute post-processing
        result = save_planning_result(llm_result, mock_context, {})

        # Result should be unchanged
        assert result == llm_result

    def test_planning_memory_includes_workflow_id(self, mock_memory_provider):
        """Test planning memories include workflow metadata."""
        from roles.core_planning import load_planning_context

        # Setup mock with workflow metadata
        mock_memories = [
            UniversalMemory(
                id="mem1",
                user_id="test_user",
                memory_type="plan",
                content="Project plan completed",
                source_role="planning",
                timestamp=1234567890.0,
                metadata={"workflow_id": "wf-123", "task_count": 5},
                tags=["project", "planning"],
            )
        ]
        mock_memory_provider.get_recent_memories.return_value = mock_memories

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        result = load_planning_context("test", mock_context, {})

        # Verify memories with metadata were loaded
        assert "tier1_memories" in result
        assert len(result["tier1_memories"]) == 1
        assert result["tier1_memories"][0].metadata["workflow_id"] == "wf-123"

    def test_planning_context_includes_user_id(self, mock_memory_provider):
        """Test context includes user_id for tool usage."""
        from roles.core_planning import load_planning_context

        mock_memory_provider.get_recent_memories.return_value = []

        mock_context = MagicMock()
        mock_context.user_id = "test_user"

        result = load_planning_context("test", mock_context, {})

        assert "user_id" in result
        assert result["user_id"] == "test_user"
