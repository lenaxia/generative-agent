"""
Tests for timer deferred workflow execution feature.

This module tests the enhanced timer system that can execute workflows
when timers expire, enabling "do X in Y time" functionality.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.intents import WorkflowIntent
from roles.core_timer import (
    TimerCreationIntent,
    TimerExpiryIntent,
    handle_heartbeat_monitoring,
    process_timer_creation_intent,
    process_timer_expiry_intent,
    set_timer,
)


class TestDeferredWorkflowCreation:
    """Test timer creation with deferred workflows."""

    def test_set_timer_with_deferred_workflow(self):
        """Test set_timer tool accepts deferred_workflow parameter."""
        result = set_timer(
            duration="10s",
            label="check weather",
            deferred_workflow="check the weather in seattle",
        )

        assert result["success"] is True
        assert "timer_id" in result
        assert "intent" in result
        assert result["intent"]["type"] == "TimerCreationIntent"
        assert result["intent"]["deferred_workflow"] == "check the weather in seattle"
        assert "will execute" in result["message"].lower()

    def test_set_timer_without_deferred_workflow(self):
        """Test set_timer still works without deferred_workflow (backward compatible)."""
        result = set_timer(duration="5m", label="coffee break")

        assert result["success"] is True
        assert result["intent"]["deferred_workflow"] == ""
        assert "will execute" not in result["message"].lower()

    def test_timer_creation_intent_with_workflow(self):
        """Test TimerCreationIntent includes deferred_workflow field."""
        intent = TimerCreationIntent(
            timer_id="timer_test123",
            duration="30s",
            duration_seconds=30,
            label="test timer",
            deferred_workflow="test workflow instruction",
            user_id="user123",
            channel_id="slack:C123",
        )

        assert intent.deferred_workflow == "test workflow instruction"
        assert intent.validate() is True


class TestDeferredWorkflowStorage:
    """Test storing deferred workflows in Redis."""

    @pytest.mark.asyncio
    async def test_process_timer_creation_stores_workflow(self):
        """Test that process_timer_creation_intent stores deferred_workflow in Redis."""
        intent = TimerCreationIntent(
            timer_id="timer_abc123",
            duration="15s",
            duration_seconds=15,
            label="weather check",
            deferred_workflow="check the weather in seattle",
            user_id="user456",
            channel_id="slack:C456",
            event_context={"user_id": "user456", "channel_id": "slack:C456"},
        )

        with (
            patch("roles.shared_tools.redis_tools.redis_write") as mock_redis_write,
            patch(
                "roles.shared_tools.redis_tools._get_redis_client"
            ) as mock_get_client,
        ):
            mock_redis_write.return_value = {"success": True}
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            await process_timer_creation_intent(intent)

            # Verify redis_write was called with deferred_workflow
            assert mock_redis_write.called
            call_args = mock_redis_write.call_args
            timer_data = call_args[0][1]  # Second argument is the data

            assert timer_data["deferred_workflow"] == "check the weather in seattle"
            assert timer_data["label"] == "weather check"
            assert timer_data["id"] == "timer_abc123"


class TestDeferredWorkflowRetrieval:
    """Test retrieving deferred workflows from Redis."""

    def test_heartbeat_monitoring_retrieves_workflow(self):
        """Test that heartbeat monitoring retrieves deferred_workflow from Redis."""
        mock_context = MagicMock()

        with (
            patch(
                "roles.core_timer._get_expired_timers_from_redis"
            ) as mock_get_expired,
            patch("roles.shared_tools.redis_tools.redis_read") as mock_redis_read,
        ):
            # Mock expired timer
            mock_get_expired.return_value = ["timer_xyz789"]

            # Mock Redis data with deferred workflow
            mock_redis_read.return_value = {
                "success": True,
                "value": {
                    "id": "timer_xyz789",
                    "duration": "20s",
                    "label": "weather check",
                    "deferred_workflow": "check the weather in portland",
                    "event_context": {"user_id": "user789", "channel_id": "slack:C789"},
                },
            }

            intents = handle_heartbeat_monitoring(None, mock_context)

            assert len(intents) == 1
            assert isinstance(intents[0], TimerExpiryIntent)
            assert intents[0].deferred_workflow == "check the weather in portland"
            assert intents[0].timer_id == "timer_xyz789"


class TestDeferredWorkflowExecution:
    """Test executing deferred workflows when timers expire."""

    @pytest.mark.asyncio
    async def test_timer_expiry_executes_workflow(self):
        """Test that timer expiry creates and processes WorkflowIntent."""
        intent = TimerExpiryIntent(
            timer_id="timer_exec123",
            original_duration="10s",
            label="weather check",
            deferred_workflow="check the weather in san francisco",
            user_id="user_exec",
            channel_id="slack:C_EXEC",
            event_context={"user_id": "user_exec", "channel_id": "slack:C_EXEC"},
        )

        # Mock the role registry and intent processor
        mock_registry = MagicMock()
        mock_intent_processor = MagicMock()
        mock_intent_processor._process_notification = AsyncMock()
        mock_intent_processor._process_workflow = AsyncMock()
        mock_registry.intent_processor = mock_intent_processor

        with patch(
            "llm_provider.role_registry.RoleRegistry.get_global_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            await process_timer_expiry_intent(intent)

            # Verify notification was sent
            assert mock_intent_processor._process_notification.called

            # Verify workflow was triggered
            assert mock_intent_processor._process_workflow.called
            workflow_call = mock_intent_processor._process_workflow.call_args[0][0]

            assert isinstance(workflow_call, WorkflowIntent)
            assert (
                workflow_call.original_instruction
                == "check the weather in san francisco"
            )
            # workflow_type is semantic type, original_instruction has the actual instruction
            assert workflow_call.workflow_type == "deferred_timer_execution"
            assert workflow_call.user_id == "user_exec"
            assert workflow_call.channel_id == "slack:C_EXEC"
            assert workflow_call.parameters["source"] == "timer_expiry"
            assert workflow_call.parameters["original_timer_id"] == "timer_exec123"

    @pytest.mark.asyncio
    async def test_timer_expiry_without_workflow_only_notifies(self):
        """Test that timer expiry without deferred_workflow only sends notification."""
        intent = TimerExpiryIntent(
            timer_id="timer_notify123",
            original_duration="5m",
            label="coffee break",
            deferred_workflow="",  # No workflow
            user_id="user_notify",
            channel_id="slack:C_NOTIFY",
        )

        mock_registry = MagicMock()
        mock_intent_processor = MagicMock()
        mock_intent_processor._process_notification = AsyncMock()
        mock_intent_processor._process_workflow = AsyncMock()
        mock_registry.intent_processor = mock_intent_processor

        with patch(
            "llm_provider.role_registry.RoleRegistry.get_global_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            await process_timer_expiry_intent(intent)

            # Verify notification was sent
            assert mock_intent_processor._process_notification.called

            # Verify workflow was NOT triggered (no deferred workflow)
            assert not mock_intent_processor._process_workflow.called


class TestDeferredWorkflowIntegration:
    """Integration tests for end-to-end deferred workflow execution."""

    @pytest.mark.asyncio
    async def test_full_deferred_workflow_cycle(self):
        """Test complete cycle: create timer with workflow -> expire -> execute workflow."""
        # Step 1: Create timer with deferred workflow
        timer_result = set_timer(
            duration="1s",  # Short duration for testing
            label="integration test",
            deferred_workflow="test workflow execution",
        )

        assert timer_result["success"] is True
        timer_id = timer_result["timer_id"]

        # Step 2: Create and process creation intent
        creation_intent = TimerCreationIntent(
            timer_id=timer_id,
            duration="1s",
            duration_seconds=1,
            label="integration test",
            deferred_workflow="test workflow execution",
            user_id="test_user",
            channel_id="test:channel",
            event_context={"user_id": "test_user", "channel_id": "test:channel"},
        )

        with (
            patch("roles.shared_tools.redis_tools.redis_write") as mock_redis_write,
            patch(
                "roles.shared_tools.redis_tools._get_redis_client"
            ) as mock_get_client,
        ):
            mock_redis_write.return_value = {"success": True}
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            await process_timer_creation_intent(creation_intent)

            # Verify timer was stored with workflow
            timer_data = mock_redis_write.call_args[0][1]
            assert timer_data["deferred_workflow"] == "test workflow execution"

        # Step 3: Simulate timer expiry
        with (
            patch(
                "roles.core_timer._get_expired_timers_from_redis"
            ) as mock_get_expired,
            patch("roles.shared_tools.redis_tools.redis_read") as mock_redis_read,
        ):
            mock_get_expired.return_value = [timer_id]
            mock_redis_read.return_value = {
                "success": True,
                "value": {
                    "id": timer_id,
                    "duration": "1s",
                    "label": "integration test",
                    "deferred_workflow": "test workflow execution",
                    "event_context": {
                        "user_id": "test_user",
                        "channel_id": "test:channel",
                    },
                },
            }

            mock_context = MagicMock()
            intents = handle_heartbeat_monitoring(None, mock_context)

            assert len(intents) == 1
            expiry_intent = intents[0]
            assert expiry_intent.deferred_workflow == "test workflow execution"

        # Step 4: Process expiry and verify workflow execution
        mock_registry = MagicMock()
        mock_intent_processor = MagicMock()
        mock_intent_processor._process_notification = AsyncMock()
        mock_intent_processor._process_workflow = AsyncMock()
        mock_registry.intent_processor = mock_intent_processor

        with patch(
            "llm_provider.role_registry.RoleRegistry.get_global_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            await process_timer_expiry_intent(expiry_intent)

            # Verify both notification and workflow execution
            assert mock_intent_processor._process_notification.called
            assert mock_intent_processor._process_workflow.called

            workflow_intent = mock_intent_processor._process_workflow.call_args[0][0]
            assert workflow_intent.original_instruction == "test workflow execution"
            assert workflow_intent.workflow_type == "deferred_timer_execution"


class TestBackwardCompatibility:
    """Test that existing timer functionality still works."""

    def test_simple_timer_without_workflow_still_works(self):
        """Test that timers without deferred_workflow work as before."""
        result = set_timer(duration="30m", label="meeting reminder")

        assert result["success"] is True
        assert result["intent"]["deferred_workflow"] == ""
        assert "Timer set for 30m (meeting reminder)" in result["message"]

    @pytest.mark.asyncio
    async def test_timer_expiry_without_workflow_backward_compatible(self):
        """Test that timer expiry without workflow behaves as before."""
        intent = TimerExpiryIntent(
            timer_id="timer_old",
            original_duration="1h",
            label="old style timer",
            user_id="user_old",
            channel_id="slack:C_OLD",
        )

        # Note: deferred_workflow defaults to "" in the dataclass
        assert intent.deferred_workflow == ""

        mock_registry = MagicMock()
        mock_intent_processor = MagicMock()
        mock_intent_processor._process_notification = AsyncMock()
        mock_intent_processor._process_workflow = AsyncMock()
        mock_registry.intent_processor = mock_intent_processor

        with patch(
            "llm_provider.role_registry.RoleRegistry.get_global_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            await process_timer_expiry_intent(intent)

            # Only notification, no workflow
            assert mock_intent_processor._process_notification.called
            assert not mock_intent_processor._process_workflow.called
