"""
Unit tests for Heartbeat health check logic.

Tests the health check logic to ensure disabled services don't cause degraded status.
"""

from unittest.mock import Mock, patch

from supervisor.heartbeat import Heartbeat


class TestHeartbeatHealthCheck:
    """Test Heartbeat health check functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_supervisor = Mock()
        self.heartbeat = Heartbeat(
            self.mock_supervisor, interval=30, health_check_interval=60
        )

    def test_healthy_components_return_healthy_status(self):
        """Test that all healthy components result in healthy overall status."""
        # Mock all health check methods to return healthy status
        with (
            patch.object(self.heartbeat, "_check_supervisor_health") as mock_supervisor,
            patch.object(
                self.heartbeat, "_check_workflow_engine_health"
            ) as mock_workflow,
            patch.object(self.heartbeat, "_check_universal_agent_health") as mock_agent,
            patch.object(self.heartbeat, "_check_mcp_health") as mock_mcp,
        ):

            mock_supervisor.return_value = {"status": "healthy"}
            mock_workflow.return_value = {"status": "healthy"}
            mock_agent.return_value = {"status": "healthy"}
            mock_mcp.return_value = {"status": "healthy"}

            self.heartbeat._perform_health_check()

            assert self.heartbeat.health_status == "healthy"

    def test_disabled_mcp_does_not_cause_degraded_status(self):
        """Test that disabled MCP service doesn't cause degraded status."""
        # Mock health checks with MCP disabled
        with (
            patch.object(self.heartbeat, "_check_supervisor_health") as mock_supervisor,
            patch.object(
                self.heartbeat, "_check_workflow_engine_health"
            ) as mock_workflow,
            patch.object(self.heartbeat, "_check_universal_agent_health") as mock_agent,
            patch.object(self.heartbeat, "_check_mcp_health") as mock_mcp,
        ):

            mock_supervisor.return_value = {"status": "healthy"}
            mock_workflow.return_value = {"status": "healthy"}
            mock_agent.return_value = {"status": "healthy"}
            mock_mcp.return_value = {
                "status": "disabled",
                "message": "MCP integration not available",
            }

            self.heartbeat._perform_health_check()

            # Should remain healthy despite disabled MCP
            assert self.heartbeat.health_status == "healthy"

    def test_unhealthy_component_causes_degraded_status(self):
        """Test that an unhealthy component causes degraded status."""
        # Mock health checks with one unhealthy component
        with (
            patch.object(self.heartbeat, "_check_supervisor_health") as mock_supervisor,
            patch.object(
                self.heartbeat, "_check_workflow_engine_health"
            ) as mock_workflow,
            patch.object(self.heartbeat, "_check_universal_agent_health") as mock_agent,
            patch.object(self.heartbeat, "_check_mcp_health") as mock_mcp,
        ):

            mock_supervisor.return_value = {"status": "healthy"}
            mock_workflow.return_value = {
                "status": "unhealthy",
                "error": "Connection failed",
            }
            mock_agent.return_value = {"status": "healthy"}
            mock_mcp.return_value = {"status": "disabled"}

            self.heartbeat._perform_health_check()

            # Should be degraded due to unhealthy workflow engine
            assert self.heartbeat.health_status == "degraded"

    def test_mixed_healthy_and_disabled_components_remain_healthy(self):
        """Test that mix of healthy and disabled components results in healthy status."""
        # Mock health checks with mix of healthy and disabled
        with (
            patch.object(self.heartbeat, "_check_supervisor_health") as mock_supervisor,
            patch.object(
                self.heartbeat, "_check_workflow_engine_health"
            ) as mock_workflow,
            patch.object(self.heartbeat, "_check_universal_agent_health") as mock_agent,
            patch.object(self.heartbeat, "_check_mcp_health") as mock_mcp,
        ):

            mock_supervisor.return_value = {"status": "healthy"}
            mock_workflow.return_value = {"status": "healthy"}
            mock_agent.return_value = {"status": "disabled"}  # Agent disabled
            mock_mcp.return_value = {"status": "disabled"}  # MCP disabled

            self.heartbeat._perform_health_check()

            # Should remain healthy with disabled services
            assert self.heartbeat.health_status == "healthy"

    def test_unexpected_status_causes_degraded(self):
        """Test that unexpected status values cause degraded status."""
        # Mock health checks with unexpected status
        with (
            patch.object(self.heartbeat, "_check_supervisor_health") as mock_supervisor,
            patch.object(
                self.heartbeat, "_check_workflow_engine_health"
            ) as mock_workflow,
            patch.object(self.heartbeat, "_check_universal_agent_health") as mock_agent,
            patch.object(self.heartbeat, "_check_mcp_health") as mock_mcp,
        ):

            mock_supervisor.return_value = {"status": "healthy"}
            mock_workflow.return_value = {"status": "unknown"}  # Unexpected status
            mock_agent.return_value = {"status": "healthy"}
            mock_mcp.return_value = {"status": "disabled"}

            self.heartbeat._perform_health_check()

            # Should be degraded due to unexpected status
            assert self.heartbeat.health_status == "degraded"

    def test_health_check_metrics_stored_correctly(self):
        """Test that health check results are stored in metrics."""
        # Mock health checks
        with (
            patch.object(self.heartbeat, "_check_supervisor_health") as mock_supervisor,
            patch.object(
                self.heartbeat, "_check_workflow_engine_health"
            ) as mock_workflow,
            patch.object(self.heartbeat, "_check_universal_agent_health") as mock_agent,
            patch.object(self.heartbeat, "_check_mcp_health") as mock_mcp,
        ):

            supervisor_result = {"status": "healthy"}
            workflow_result = {"status": "healthy"}
            agent_result = {"status": "healthy"}
            mcp_result = {
                "status": "disabled",
                "message": "MCP integration not available",
            }

            mock_supervisor.return_value = supervisor_result
            mock_workflow.return_value = workflow_result
            mock_agent.return_value = agent_result
            mock_mcp.return_value = mcp_result

            self.heartbeat._perform_health_check()

            # Verify metrics are stored correctly
            health_check_metrics = self.heartbeat.metrics.get("health_check", {})
            assert health_check_metrics["overall_status"] == "healthy"
            assert health_check_metrics["components"]["supervisor"] == supervisor_result
            assert (
                health_check_metrics["components"]["workflow_engine"] == workflow_result
            )
            assert health_check_metrics["components"]["universal_agent"] == agent_result
            assert health_check_metrics["components"]["mcp_servers"] == mcp_result

    def test_get_health_status_returns_correct_format(self):
        """Test that get_health_status returns properly formatted data."""
        # Set up some test metrics
        self.heartbeat.health_status = "healthy"
        self.heartbeat.metrics = {
            "heartbeat": {"last_heartbeat": 1234567890},
            "system": {"cpu_percent": 10.5},
        }

        with patch("time.time", return_value=1234567900):
            health_status = self.heartbeat.get_health_status()

            assert health_status["overall_status"] == "healthy"
            assert "metrics" in health_status
            assert "uptime" in health_status
            assert "last_heartbeat" in health_status

    def test_is_healthy_method(self):
        """Test the is_healthy convenience method."""
        # Test healthy status
        self.heartbeat.health_status = "healthy"
        assert self.heartbeat.is_healthy() is True

        # Test degraded status
        self.heartbeat.health_status = "degraded"
        assert self.heartbeat.is_healthy() is False

        # Test unknown status
        self.heartbeat.health_status = "unknown"
        assert self.heartbeat.is_healthy() is False
