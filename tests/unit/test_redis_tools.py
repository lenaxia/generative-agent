"""Unit tests for Redis shared tools.

Tests Redis operations with automatic key prefixing, TTL support,
and both synchronous and asynchronous operations.
"""

import json
import os

# Import the Redis tools
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from roles.shared_tools.redis_tools import (
    _get_calling_role,
    _prefix_key,
    _unprefix_key,
    redis_clear_role_data,
    redis_delete,
    redis_exists,
    redis_get_keys,
    redis_health_check,
    redis_read,
    redis_read_async,
    redis_write,
    redis_write_async,
)


class TestRedisKeyPrefixing:
    """Test automatic key prefixing functionality."""

    def test_prefix_key_with_role(self):
        """Test key prefixing with explicit role."""
        result = _prefix_key("test_key", "weather")
        assert result == "weather:test_key"

    def test_prefix_key_auto_detect(self):
        """Test key prefixing with auto-detection."""
        with patch(
            "roles.shared_tools.redis_tools._get_calling_role", return_value="timer"
        ):
            result = _prefix_key("session_data")
            assert result == "timer:session_data"

    def test_unprefix_key_with_role(self):
        """Test key unprefixing with explicit role."""
        result = _unprefix_key("weather:test_key", "weather")
        assert result == "test_key"

    def test_unprefix_key_auto_detect(self):
        """Test key unprefixing with auto-detection."""
        with patch(
            "roles.shared_tools.redis_tools._get_calling_role", return_value="timer"
        ):
            result = _unprefix_key("timer:session_data")
            assert result == "session_data"

    def test_unprefix_key_no_prefix(self):
        """Test unprefixing key that doesn't have the expected prefix."""
        result = _unprefix_key("other:test_key", "weather")
        assert result == "other:test_key"  # Should return unchanged

    def test_get_calling_role_fallback(self):
        """Test that _get_calling_role returns 'unknown' when no role is detected."""
        # This will be 'unknown' since we're not in a role context
        result = _get_calling_role()
        assert isinstance(result, str)
        assert len(result) > 0


class TestRedisSyncOperations:
    """Test synchronous Redis operations."""

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_write_string(self, mock_get_role, mock_get_client):
        """Test writing string data to Redis."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.set.return_value = True

        result = redis_write("test_key", "test_value")

        assert result["success"] is True
        assert result["key"] == "test_key"
        assert result["prefixed_key"] == "test_role:test_key"
        mock_client.set.assert_called_once_with("test_role:test_key", "test_value")

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_write_with_ttl(self, mock_get_role, mock_get_client):
        """Test writing data with TTL."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.setex.return_value = True

        result = redis_write("test_key", "test_value", ttl=3600)

        assert result["success"] is True
        assert result["ttl"] == 3600
        mock_client.setex.assert_called_once_with(
            "test_role:test_key", 3600, "test_value"
        )

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_write_dict(self, mock_get_role, mock_get_client):
        """Test writing dictionary data to Redis."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.set.return_value = True

        test_data = {"user_id": 123, "name": "test"}
        result = redis_write("user_data", test_data)

        assert result["success"] is True
        expected_json = json.dumps(test_data)
        mock_client.set.assert_called_once_with("test_role:user_data", expected_json)

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_write_error(self, mock_get_role, mock_get_client):
        """Test Redis write error handling."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.set.side_effect = Exception("Connection failed")

        result = redis_write("test_key", "test_value")

        assert result["success"] is False
        assert "Connection failed" in result["error"]

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_read_success(self, mock_get_role, mock_get_client):
        """Test successful Redis read."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = "test_value"
        mock_client.ttl.return_value = 3600

        result = redis_read("test_key")

        assert result["success"] is True
        assert result["value"] == "test_value"
        assert result["ttl"] == 3600
        mock_client.get.assert_called_once_with("test_role:test_key")

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_read_json(self, mock_get_role, mock_get_client):
        """Test reading JSON data from Redis."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        test_data = {"user_id": 123, "name": "test"}
        mock_client.get.return_value = json.dumps(test_data)
        mock_client.ttl.return_value = -1

        result = redis_read("user_data")

        assert result["success"] is True
        assert result["value"] == test_data
        assert result["ttl"] is None  # -1 TTL becomes None

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_read_not_found(self, mock_get_role, mock_get_client):
        """Test reading non-existent key."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = None

        result = redis_read("nonexistent_key")

        assert result["success"] is False
        assert result["value"] is None
        assert "not found" in result["message"].lower()

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_get_keys(self, mock_get_role, mock_get_client):
        """Test getting keys for a role."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.keys.return_value = [
            "test_role:session:123",
            "test_role:session:456",
            "test_role:user:789",
        ]

        result = redis_get_keys("session:*")

        assert result["success"] is True
        assert result["count"] == 3
        assert "session:123" in result["keys"]
        assert "session:456" in result["keys"]
        assert "user:789" in result["keys"]
        mock_client.keys.assert_called_once_with("test_role:session:*")

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_delete(self, mock_get_role, mock_get_client):
        """Test deleting a key."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.delete.return_value = 1

        result = redis_delete("test_key")

        assert result["success"] is True
        assert result["deleted"] is True
        mock_client.delete.assert_called_once_with("test_role:test_key")

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_exists(self, mock_get_role, mock_get_client):
        """Test checking if key exists."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.exists.return_value = 1

        result = redis_exists("test_key")

        assert result["success"] is True
        assert result["exists"] is True
        mock_client.exists.assert_called_once_with("test_role:test_key")

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_clear_role_data(self, mock_get_role, mock_get_client):
        """Test clearing all data for a role."""
        mock_get_role.return_value = "test_role"
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.keys.return_value = ["test_role:key1", "test_role:key2"]
        mock_client.delete.return_value = 2

        result = redis_clear_role_data()

        assert result["success"] is True
        assert result["deleted_count"] == 2
        mock_client.keys.assert_called_once_with("test_role:*")
        mock_client.delete.assert_called_once_with("test_role:key1", "test_role:key2")

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    def test_redis_health_check(self, mock_get_client):
        """Test Redis health check."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.set.return_value = True
        mock_client.get.return_value = "test"
        mock_client.delete.return_value = 1
        mock_client.info.return_value = {
            "redis_version": "6.2.0",
            "used_memory_human": "1.2M",
            "connected_clients": 5,
        }

        result = redis_health_check()

        assert result["success"] is True
        assert result["connected"] is True
        assert result["redis_version"] == "6.2.0"


class TestRedisAsyncOperations:
    """Test asynchronous Redis operations."""

    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_write_async_returns_immediately(self, mock_get_role):
        """Test that async write returns immediately."""
        mock_get_role.return_value = "test_role"

        result = redis_write_async("test_key", "test_value")

        # Should return immediately with success
        assert result["success"] is True
        assert result["async"] is True
        assert result["key"] == "test_key"
        assert "started" in result["message"].lower()

    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_read_async_returns_immediately(self, mock_get_role):
        """Test that async read returns immediately."""
        mock_get_role.return_value = "test_role"

        callback = Mock()
        result = redis_read_async("test_key", callback=callback)

        # Should return immediately with success
        assert result["success"] is True
        assert result["async"] is True
        assert result["callback"] is True
        assert result["key"] == "test_key"

    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_read_async_no_callback(self, mock_get_role):
        """Test async read without callback."""
        mock_get_role.return_value = "test_role"

        result = redis_read_async("test_key")

        assert result["success"] is True
        assert result["callback"] is False


class TestRedisConnectionHandling:
    """Test Redis connection and error handling."""

    @patch("roles.shared_tools.redis_tools.REDIS_AVAILABLE", False)
    def test_redis_not_available(self):
        """Test behavior when Redis is not installed."""
        result = redis_write("test_key", "test_value")
        assert result["success"] is False
        assert "Redis not available" in result["error"]

    @patch("roles.shared_tools.redis_tools._get_redis_client")
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_connection_error_handling(self, mock_get_role, mock_get_client):
        """Test handling of Redis connection errors."""
        mock_get_role.return_value = "test_role"
        mock_get_client.side_effect = Exception("Connection refused")

        result = redis_write("test_key", "test_value")

        assert result["success"] is False
        assert "Connection refused" in result["error"]


class TestRedisConfiguration:
    """Test Redis configuration handling."""

    @patch.dict(
        os.environ,
        {
            "REDIS_HOST": "test-host",
            "REDIS_PORT": "6380",
            "REDIS_DB": "1",
            "REDIS_PASSWORD": "test-password",
        },
    )
    def test_redis_config_from_env(self):
        """Test Redis configuration from environment variables."""
        from roles.shared_tools.redis_tools import _get_redis_config

        config = _get_redis_config()

        assert config["host"] == "test-host"
        assert config["port"] == 6380
        assert config["db"] == 1
        assert config["password"] == "test-password"


class TestRedisIntegration:
    """Integration tests for Redis tools (requires actual Redis connection)."""

    @pytest.mark.integration
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_full_redis_workflow(self, mock_get_role):
        """Test complete Redis workflow with real operations."""
        mock_get_role.return_value = "integration_test"

        # Skip if Redis is not available
        try:
            health = redis_health_check()
            if not health["success"]:
                pytest.skip("Redis not available for integration test")
        except Exception:
            pytest.skip("Redis not available for integration test")

        # Test data
        test_key = "integration_test_key"
        test_data = {"timestamp": "2024-01-01", "test": True}

        try:
            # Write data
            write_result = redis_write(test_key, test_data, ttl=60)
            assert write_result["success"] is True

            # Read data back
            read_result = redis_read(test_key)
            assert read_result["success"] is True
            assert read_result["value"] == test_data
            assert read_result["ttl"] is not None

            # Check existence
            exists_result = redis_exists(test_key)
            assert exists_result["success"] is True
            assert exists_result["exists"] is True

            # Get keys
            keys_result = redis_get_keys("integration_*")
            assert keys_result["success"] is True
            assert test_key in keys_result["keys"]

            # Delete key
            delete_result = redis_delete(test_key)
            assert delete_result["success"] is True
            assert delete_result["deleted"] is True

            # Verify deletion
            read_after_delete = redis_read(test_key)
            assert read_after_delete["success"] is False

        finally:
            # Cleanup
            redis_clear_role_data("integration_test")


if __name__ == "__main__":
    # Run tests with timeout to prevent hangs
    pytest.main([__file__, "-v", "--timeout=30"])
