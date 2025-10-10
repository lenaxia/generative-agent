"""Integration tests for Docker Redis setup."""

import json
import subprocess
import time
from unittest.mock import patch

import pytest
import redis

from roles.shared_tools.redis_tools import redis_health_check, redis_read, redis_write


class TestDockerRedisSetup:
    """Test Docker Redis setup and integration."""

    @pytest.fixture(scope="class")
    def docker_redis_available(self):
        """Check if Docker Redis container is available."""
        try:
            # Try to connect to Redis on Docker port
            client = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=5)
            client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            pytest.skip("Docker Redis container not available")

    def test_docker_compose_file_exists(self):
        """Test that docker-compose.yml exists and is valid."""
        import os

        import yaml

        compose_file = "docker-compose.yml"
        assert os.path.exists(compose_file), "docker-compose.yml not found"

        with open(compose_file) as f:
            compose_config = yaml.safe_load(f)

        # Verify Redis service exists
        assert "services" in compose_config
        assert "redis" in compose_config["services"]

        redis_service = compose_config["services"]["redis"]
        assert redis_service["image"] == "redis:7-alpine"
        assert "6379:6379" in redis_service["ports"]

    def test_redis_config_file_exists(self):
        """Test that Redis configuration file exists."""
        import os

        redis_config = "docker/redis.conf"
        assert os.path.exists(redis_config), "docker/redis.conf not found"

        with open(redis_config) as f:
            config_content = f.read()

        # Check for key configuration settings
        assert "port 6379" in config_content
        assert "maxmemory 256mb" in config_content
        assert "appendonly yes" in config_content

    @pytest.mark.integration
    def test_redis_container_health(self, docker_redis_available):
        """Test Redis container health and connectivity."""
        if not docker_redis_available:
            return

        # Test direct Redis connection
        client = redis.Redis(host="localhost", port=6379, db=0)

        # Test ping
        assert client.ping() is True

        # Test basic operations
        test_key = "test:docker:health"
        test_value = "docker_test_value"

        client.set(test_key, test_value)
        retrieved_value = client.get(test_key)
        assert retrieved_value.decode() == test_value

        # Cleanup
        client.delete(test_key)

    @pytest.mark.integration
    def test_application_redis_integration(self, docker_redis_available):
        """Test application Redis tools with Docker Redis."""
        if not docker_redis_available:
            return

        # Test health check
        health_result = redis_health_check()
        assert health_result["success"] is True
        assert health_result["connected"] is True
        assert "redis_version" in health_result

    @pytest.mark.integration
    @patch("roles.shared_tools.redis_tools._get_calling_role")
    def test_redis_tools_with_docker(self, mock_get_role, docker_redis_available):
        """Test Redis tools functionality with Docker Redis."""
        if not docker_redis_available:
            return

        mock_get_role.return_value = "docker_test"

        # Test write and read
        test_key = "integration_test"
        test_data = {
            "message": "Docker Redis integration test",
            "timestamp": time.time(),
            "test_id": "docker_integration_001",
        }

        # Write data
        write_result = redis_write(test_key, test_data, ttl=60)
        assert write_result["success"] is True

        # Read data back
        read_result = redis_read(test_key)
        assert read_result["success"] is True
        assert read_result["value"]["message"] == test_data["message"]
        assert read_result["value"]["test_id"] == test_data["test_id"]

    def test_dev_setup_script_exists(self):
        """Test that development setup script exists and is executable."""
        import os
        import stat

        script_path = "scripts/dev-setup.sh"
        assert os.path.exists(script_path), "dev-setup.sh script not found"

        # Check if script is executable
        file_stat = os.stat(script_path)
        assert file_stat.st_mode & stat.S_IEXEC, "dev-setup.sh is not executable"

    @pytest.mark.integration
    def test_docker_compose_redis_startup(self):
        """Test Docker Compose Redis service startup."""
        try:
            # Check if docker-compose is available
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                pytest.skip("Docker Compose not available")

            # Test docker-compose config validation
            result = subprocess.run(
                ["docker-compose", "config"], capture_output=True, text=True, timeout=30
            )
            assert (
                result.returncode == 0
            ), f"Docker Compose config invalid: {result.stderr}"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker Compose not available or timeout")

    @pytest.mark.integration
    def test_redis_commander_service(self):
        """Test Redis Commander service configuration."""
        import yaml

        with open("docker-compose.yml") as f:
            compose_config = yaml.safe_load(f)

        # Check Redis Commander service
        assert "redis-commander" in compose_config["services"]

        commander_service = compose_config["services"]["redis-commander"]
        assert "8081:8081" in commander_service["ports"]
        assert commander_service["profiles"] == ["tools"]

    def test_config_yaml_redis_section(self):
        """Test that config.yaml has proper Redis configuration."""
        import yaml

        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Check Redis configuration exists
        assert "redis" in config
        redis_config = config["redis"]

        assert redis_config["enabled"] is True
        assert "connection" in redis_config

        connection_config = redis_config["connection"]
        assert connection_config["host"] == "localhost"
        assert connection_config["port"] == 6379
        assert connection_config["db"] == 0

    @pytest.mark.integration
    def test_timer_manager_with_docker_redis(self, docker_redis_available):
        """Test TimerManager integration with Docker Redis."""
        if not docker_redis_available:
            return

        from roles.timer.lifecycle import TimerManager

        # Create TimerManager instance
        timer_manager = TimerManager(redis_host="localhost", redis_port=6379)

        # Test Redis connection
        assert timer_manager.redis.ping() is True

    @pytest.mark.integration
    def test_redis_performance_with_docker(self, docker_redis_available):
        """Test Redis performance with Docker setup."""
        if not docker_redis_available:
            return

        import time

        client = redis.Redis(host="localhost", port=6379, db=0)

        # Performance test - write/read operations
        start_time = time.time()

        for i in range(100):
            test_key = f"perf_test:{i}"
            test_value = f"performance_test_value_{i}"
            client.set(test_key, test_value)
            retrieved = client.get(test_key)
            assert retrieved.decode() == test_value
            client.delete(test_key)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete 100 operations in reasonable time (< 5 seconds)
        assert duration < 5.0, f"Redis operations too slow: {duration}s for 100 ops"

    def test_makefile_docker_commands(self):
        """Test that Makefile includes Docker-related commands."""
        import os

        if not os.path.exists("Makefile"):
            pytest.skip("Makefile not found")

        with open("Makefile") as f:
            makefile_content = f.read()

        # Should have docker-related targets
        expected_targets = ["docker", "redis", "dev-setup"]
        found_targets = []

        for target in expected_targets:
            if (
                f"{target}:" in makefile_content
                or f".PHONY: {target}" in makefile_content
            ):
                found_targets.append(target)

        # At least some Docker-related functionality should be present
        assert len(found_targets) > 0, "No Docker-related Makefile targets found"


class TestDockerRedisConfiguration:
    """Test Docker Redis configuration and settings."""

    def test_redis_conf_security_settings(self):
        """Test Redis configuration security settings."""
        with open("docker/redis.conf") as f:
            config_content = f.read()

        # Check security settings
        assert "protected-mode no" in config_content  # OK for development
        assert 'requirepass ""' in config_content  # No password for dev

    def test_redis_conf_performance_settings(self):
        """Test Redis configuration performance settings."""
        with open("docker/redis.conf") as f:
            config_content = f.read()

        # Check performance settings
        assert "maxmemory 256mb" in config_content
        assert "maxmemory-policy allkeys-lru" in config_content
        assert "appendonly yes" in config_content

    def test_redis_conf_persistence_settings(self):
        """Test Redis configuration persistence settings."""
        with open("docker/redis.conf") as f:
            config_content = f.read()

        # Check persistence settings
        assert "save 900 1" in config_content
        assert "save 300 10" in config_content
        assert "save 60 10000" in config_content
        assert "dir /data" in config_content
