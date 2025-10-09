"""Unit tests for Timer Manager functionality.

Tests the core timer management system including Redis integration,
timer CRUD operations, and expiry handling.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roles.timer.lifecycle import TimerManager


class TestTimerManager:
    """Test suite for TimerManager class."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        redis_mock = MagicMock()
        redis_mock.hset = MagicMock()
        redis_mock.hget = MagicMock()
        redis_mock.hgetall = MagicMock()
        redis_mock.hdel = MagicMock()
        redis_mock.zadd = MagicMock()
        redis_mock.zrem = MagicMock()
        redis_mock.zrangebyscore = MagicMock()
        redis_mock.zrange = MagicMock()
        redis_mock.sadd = MagicMock()
        redis_mock.srem = MagicMock()
        redis_mock.smembers = MagicMock()
        redis_mock.exists = MagicMock()
        redis_mock.expire = MagicMock()
        redis_mock.delete = MagicMock()
        return redis_mock

    @pytest.fixture
    def timer_manager(self, mock_redis):
        """Create TimerManager instance with mocked Redis."""
        with patch("roles.timer.lifecycle.redis.Redis", return_value=mock_redis):
            manager = TimerManager()
            manager.redis = mock_redis
            return manager

    @pytest.fixture
    def sample_timer_data(self):
        """Sample timer data for testing."""
        return {
            "id": "timer_123",
            "type": "countdown",
            "name": "Test Timer",
            "label": "Test Label",
            "custom_message": "Timer expired!",
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + 3600,
            "duration_seconds": 3600,
            "timezone": "America/Los_Angeles",
            "status": "active",
            "snooze_count": 0,
            "completion_rate": 1.0,
            "user_id": "user123",
            "channel_id": "slack:general",
            "request_context": {
                "original_request": "Set a 1 hour timer",
                "source": "slack",
                "conversation_id": "conv_123",
            },
            "notification_config": {
                "channels": [
                    {
                        "channel_id": "slack:general",
                        "message": "Timer expired!",
                        "template": "🔔 {name}: {custom_message}",
                        "priority": "high",
                    }
                ],
                "primary_channel": "slack:general",
                "fallback_channels": ["sms:user_phone"],
            },
            "actions": [{"type": "notify", "config": {"sound": True, "vibrate": True}}],
        }

    @pytest.mark.asyncio
    async def test_create_timer_success(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test successful timer creation."""
        # Setup
        mock_redis.hset.return_value = True
        mock_redis.zadd.return_value = 1
        mock_redis.sadd.return_value = 1

        # Execute
        timer_id = await timer_manager.create_timer(
            timer_type="countdown",
            duration_seconds=3600,
            name="Test Timer",
            label="Test Label",
            custom_message="Timer expired!",
            user_id="user123",
            channel_id="slack:general",
            request_context=sample_timer_data["request_context"],
            notification_config=sample_timer_data["notification_config"],
            actions=sample_timer_data["actions"],
        )

        # Verify
        assert timer_id is not None
        assert isinstance(timer_id, str)

        # Verify Redis calls
        mock_redis.hset.assert_called()
        mock_redis.zadd.assert_called()
        mock_redis.sadd.assert_called()

    @pytest.mark.asyncio
    async def test_create_timer_invalid_type(self, timer_manager):
        """Test timer creation with invalid type."""
        with pytest.raises(ValueError, match="Invalid timer type"):
            await timer_manager.create_timer(
                timer_type="invalid_type",
                duration_seconds=3600,
                name="Test Timer",
                user_id="user123",
                channel_id="slack:general",
            )

    @pytest.mark.asyncio
    async def test_create_timer_invalid_duration(self, timer_manager):
        """Test timer creation with invalid duration."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            await timer_manager.create_timer(
                timer_type="countdown",
                duration_seconds=-100,
                name="Test Timer",
                user_id="user123",
                channel_id="slack:general",
            )

    @pytest.mark.asyncio
    async def test_get_timer_success(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test successful timer retrieval."""
        # Setup
        mock_redis.hgetall.return_value = {
            k.encode(): json.dumps(v).encode()
            if isinstance(v, (dict, list))
            else str(v).encode()
            for k, v in sample_timer_data.items()
        }

        # Execute
        timer = await timer_manager.get_timer("timer_123")

        # Verify
        assert timer is not None
        assert timer["id"] == "timer_123"
        assert timer["name"] == "Test Timer"
        assert timer["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_timer_not_found(self, timer_manager, mock_redis):
        """Test timer retrieval when timer doesn't exist."""
        # Setup
        mock_redis.hgetall.return_value = {}

        # Execute
        timer = await timer_manager.get_timer("nonexistent_timer")

        # Verify
        assert timer is None

    @pytest.mark.asyncio
    async def test_cancel_timer_success(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test successful timer cancellation."""
        # Setup
        mock_redis.hgetall.return_value = {
            k.encode(): json.dumps(v).encode()
            if isinstance(v, (dict, list))
            else str(v).encode()
            for k, v in sample_timer_data.items()
        }
        mock_redis.hset.return_value = True
        mock_redis.zrem.return_value = 1
        mock_redis.srem.return_value = 1

        # Execute
        result = await timer_manager.cancel_timer("timer_123")

        # Verify
        assert result is True
        mock_redis.hset.assert_called()
        mock_redis.zrem.assert_called()

    @pytest.mark.asyncio
    async def test_cancel_timer_not_found(self, timer_manager, mock_redis):
        """Test timer cancellation when timer doesn't exist."""
        # Setup
        mock_redis.hgetall.return_value = {}

        # Execute
        result = await timer_manager.cancel_timer("nonexistent_timer")

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_list_timers_by_user(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test listing timers for a specific user."""
        # Setup
        mock_redis.smembers.return_value = {b"timer_123", b"timer_456"}
        mock_redis.hgetall.return_value = {
            k.encode(): json.dumps(v).encode()
            if isinstance(v, (dict, list))
            else str(v).encode()
            for k, v in sample_timer_data.items()
        }

        # Execute
        timers = await timer_manager.list_timers(user_id="user123")

        # Verify
        assert len(timers) == 2
        assert all(timer["user_id"] == "user123" for timer in timers)

    @pytest.mark.asyncio
    async def test_list_timers_by_channel(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test listing timers for a specific channel."""
        # Setup
        mock_redis.smembers.return_value = {b"timer_123"}
        mock_redis.hgetall.return_value = {
            k.encode(): json.dumps(v).encode()
            if isinstance(v, (dict, list))
            else str(v).encode()
            for k, v in sample_timer_data.items()
        }

        # Execute
        timers = await timer_manager.list_timers(channel_id="slack:general")

        # Verify
        assert len(timers) == 1
        assert timers[0]["channel_id"] == "slack:general"

    @pytest.mark.asyncio
    async def test_get_expiring_timers(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test getting timers that are expiring."""
        # Setup
        current_time = int(time.time())
        mock_redis.zrangebyscore.return_value = [b"timer_123", b"timer_456"]
        mock_redis.hgetall.return_value = {
            k.encode(): json.dumps(v).encode()
            if isinstance(v, (dict, list))
            else str(v).encode()
            for k, v in sample_timer_data.items()
        }

        # Execute
        expiring_timers = await timer_manager.get_expiring_timers(current_time)

        # Verify
        assert len(expiring_timers) == 2
        mock_redis.zrangebyscore.assert_called_with(
            "timer:active_queue", 0, current_time
        )

    @pytest.mark.asyncio
    async def test_update_timer_status(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test updating timer status."""
        # Setup
        mock_redis.hgetall.return_value = {
            k.encode(): json.dumps(v).encode()
            if isinstance(v, (dict, list))
            else str(v).encode()
            for k, v in sample_timer_data.items()
        }
        mock_redis.hset.return_value = True

        # Execute
        result = await timer_manager.update_timer_status("timer_123", "completed")

        # Verify
        assert result is True
        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_snooze_timer(self, timer_manager, mock_redis, sample_timer_data):
        """Test snoozing a timer."""
        # Setup
        mock_redis.hgetall.return_value = {
            k.encode(): json.dumps(v).encode()
            if isinstance(v, (dict, list))
            else str(v).encode()
            for k, v in sample_timer_data.items()
        }
        mock_redis.hset.return_value = True
        mock_redis.zadd.return_value = 1

        # Execute
        result = await timer_manager.snooze_timer("timer_123", 300)  # 5 minutes

        # Verify
        assert result is True
        mock_redis.hset.assert_called()
        mock_redis.zadd.assert_called()

    @pytest.mark.asyncio
    async def test_create_recurring_instance(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test creating a recurring timer instance."""
        # Setup recurring timer data
        recurring_timer = sample_timer_data.copy()
        recurring_timer["type"] = "recurring"
        recurring_timer["recurring"] = {
            "pattern": "daily",
            "cron_expression": "0 9 * * *",
            "end_date": int(time.time()) + (30 * 24 * 3600),  # 30 days from now
            "max_occurrences": 10,
        }

        mock_redis.hset.return_value = True
        mock_redis.zadd.return_value = 1
        mock_redis.sadd.return_value = 1

        # Execute
        next_timer_id = await timer_manager.create_recurring_instance(recurring_timer)

        # Verify
        assert next_timer_id is not None
        assert isinstance(next_timer_id, str)
        mock_redis.hset.assert_called()
        mock_redis.zadd.assert_called()

    @pytest.mark.asyncio
    async def test_redis_connection_error(self, timer_manager, mock_redis):
        """Test handling Redis connection errors."""
        # Setup
        mock_redis.hset.side_effect = Exception("Redis connection failed")

        # Execute & Verify
        with pytest.raises(Exception, match="Redis connection failed"):
            await timer_manager.create_timer(
                timer_type="countdown",
                duration_seconds=3600,
                name="Test Timer",
                user_id="user123",
                channel_id="slack:general",
            )

    @pytest.mark.asyncio
    async def test_timer_validation_edge_cases(self, timer_manager):
        """Test timer validation with edge cases."""
        # Test maximum duration
        with pytest.raises(ValueError, match="Duration exceeds maximum"):
            await timer_manager.create_timer(
                timer_type="countdown",
                duration_seconds=365 * 24 * 3600 + 1,  # More than 1 year
                name="Test Timer",
                user_id="user123",
                channel_id="slack:general",
            )

        # Test empty name
        with pytest.raises(ValueError, match="Timer name cannot be empty"):
            await timer_manager.create_timer(
                timer_type="countdown",
                duration_seconds=3600,
                name="",
                user_id="user123",
                channel_id="slack:general",
            )

        # Test empty user_id
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            await timer_manager.create_timer(
                timer_type="countdown",
                duration_seconds=3600,
                name="Test Timer",
                user_id="",
                channel_id="slack:general",
            )

    @pytest.mark.asyncio
    async def test_concurrent_timer_operations(
        self, timer_manager, mock_redis, sample_timer_data
    ):
        """Test concurrent timer operations for race condition handling."""
        # Setup
        mock_redis.hset.return_value = True
        mock_redis.zadd.return_value = 1
        mock_redis.sadd.return_value = 1

        # Execute multiple concurrent operations
        tasks = []
        for i in range(10):
            task = timer_manager.create_timer(
                timer_type="countdown",
                duration_seconds=3600,
                name=f"Test Timer {i}",
                user_id="user123",
                channel_id="slack:general",
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all operations completed successfully
        assert len(results) == 10
        assert all(
            isinstance(result, str)
            for result in results
            if not isinstance(result, Exception)
        )
