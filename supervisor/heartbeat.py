import logging
import threading
import time
from typing import Any, Dict

import psutil

logger = logging.getLogger(__name__)


class Heartbeat:
    """
    Heartbeat service for monitoring system health and performing periodic maintenance.

    Integrates with the new WorkflowEngine architecture to provide:
    - System health monitoring
    - Workflow progress tracking
    - Resource usage monitoring
    - Automatic cleanup and maintenance
    - Performance metrics collection
    """

    def __init__(self, supervisor, interval: int = 30, health_check_interval: int = 60):
        """
        Initialize Heartbeat service.

        Args:
            supervisor: Supervisor instance to monitor
            interval: Heartbeat interval in seconds (default: 30)
            health_check_interval: Health check interval in seconds (default: 60)
        """
        self.supervisor = supervisor
        self.interval = interval
        self.health_check_interval = health_check_interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.last_health_check = 0
        self.health_status = "unknown"
        self.metrics = {}

    def start(self):
        """Start the heartbeat service"""
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            logger.info(f"Heartbeat started with {self.interval}s interval")
        else:
            logger.warning("Heartbeat already running")

    def stop(self):
        """Stop the heartbeat service"""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("Heartbeat stopped")

    def run(self):
        """Main heartbeat loop"""
        logger.info("Heartbeat service running")

        while not self.stop_event.is_set():
            try:
                # Perform heartbeat operations
                self._perform_heartbeat()

                # Perform health check if interval elapsed
                current_time = time.time()
                if current_time - self.last_health_check >= self.health_check_interval:
                    self._perform_health_check()
                    self.last_health_check = current_time

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            # Wait for next heartbeat
            self.stop_event.wait(self.interval)

    def _perform_heartbeat(self):
        """Perform regular heartbeat operations"""
        try:
            # Monitor WorkflowEngine status
            self._monitor_workflow_engine()

            # Monitor system resources
            self._monitor_system_resources()

            # Perform maintenance tasks
            self._perform_maintenance()

            # Update metrics
            self._update_metrics()

        except Exception as e:
            logger.error(f"Heartbeat operation failed: {e}")

    def _monitor_workflow_engine(self):
        """Monitor WorkflowEngine health and performance"""
        try:
            if hasattr(self.supervisor, "workflow_engine"):
                workflow_engine = self.supervisor.workflow_engine

                # Get workflow metrics
                metrics = workflow_engine.get_workflow_metrics()

                # Log workflow status
                active_count = metrics.get("active_workflows", 0)
                queue_size = metrics.get("queue_size", 0)

                if active_count > 50:
                    logger.warning(f"High active workflow count: {active_count}")

                if queue_size > 100:
                    logger.warning(f"Large task queue: {queue_size}")

                # Store metrics for health check
                self.metrics["workflow_engine"] = {
                    "active_workflows": active_count,
                    "queue_size": queue_size,
                    "state": metrics.get("state", "unknown"),
                    "last_updated": time.time(),
                }

        except Exception as e:
            logger.error(f"WorkflowEngine monitoring failed: {e}")
            self.metrics["workflow_engine"] = {"status": "error", "error": str(e)}

    def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Log warnings for high usage
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent}%")

            if memory_percent > 85:
                logger.warning(f"High memory usage: {memory_percent}%")

            if disk_percent > 90:
                logger.warning(f"High disk usage: {disk_percent}%")

            # Store metrics
            self.metrics["system"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "memory_available": memory.available,
                "disk_free": disk.free,
                "last_updated": time.time(),
            }

        except Exception as e:
            logger.error(f"System resource monitoring failed: {e}")
            self.metrics["system"] = {"status": "error", "error": str(e)}

    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        try:
            # Clean up old workflow data
            self._cleanup_old_workflows()

            # Clean up old log files
            self._cleanup_old_logs()

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

        except Exception as e:
            logger.error(f"Maintenance task failed: {e}")

    def _cleanup_old_workflows(self):
        """Clean up completed workflows older than retention period"""
        try:
            if hasattr(self.supervisor, "workflow_engine"):
                workflow_engine = self.supervisor.workflow_engine

                # Get completed workflows older than 24 hours
                cutoff_time = time.time() - (24 * 3600)  # 24 hours ago

                workflows_to_cleanup = []
                for workflow_id, context in workflow_engine.active_workflows.items():
                    if (
                        hasattr(context, "completed_at")
                        and context.completed_at
                        and context.completed_at < cutoff_time
                    ):
                        workflows_to_cleanup.append(workflow_id)

                # Clean up old workflows
                for workflow_id in workflows_to_cleanup:
                    try:
                        del workflow_engine.active_workflows[workflow_id]
                        logger.debug(f"Cleaned up old workflow: {workflow_id}")
                    except Exception as e:
                        logger.error(f"Failed to cleanup workflow {workflow_id}: {e}")

                if workflows_to_cleanup:
                    logger.info(f"Cleaned up {len(workflows_to_cleanup)} old workflows")

        except Exception as e:
            logger.error(f"Workflow cleanup failed: {e}")

    def _cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            import glob
            import os

            log_dir = "logs"
            if os.path.exists(log_dir):
                # Remove log files older than 30 days
                cutoff_time = time.time() - (30 * 24 * 3600)  # 30 days ago

                log_files = glob.glob(os.path.join(log_dir, "*.log*"))
                cleaned_count = 0

                for log_file in log_files:
                    try:
                        if os.path.getmtime(log_file) < cutoff_time:
                            os.remove(log_file)
                            cleaned_count += 1
                    except Exception as e:
                        logger.error(f"Failed to remove log file {log_file}: {e}")

                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old log files")

        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoint files"""
        try:
            import glob
            import os

            checkpoint_dir = "checkpoints"
            if os.path.exists(checkpoint_dir):
                # Remove checkpoint files older than 7 days
                cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago

                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.json"))
                cleaned_count = 0

                for checkpoint_file in checkpoint_files:
                    try:
                        if os.path.getmtime(checkpoint_file) < cutoff_time:
                            os.remove(checkpoint_file)
                            cleaned_count += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to remove checkpoint file {checkpoint_file}: {e}"
                        )

                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old checkpoint files")

        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")

    def _update_metrics(self):
        """Update heartbeat metrics"""
        self.metrics["heartbeat"] = {
            "last_heartbeat": time.time(),
            "uptime": time.time() - getattr(self, "start_time", time.time()),
            "health_status": self.health_status,
            "thread_alive": self.thread.is_alive(),
        }

    def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            health_results = {}

            # Check Supervisor status
            supervisor_status = self._check_supervisor_health()
            health_results["supervisor"] = supervisor_status

            # Check WorkflowEngine health
            workflow_engine_status = self._check_workflow_engine_health()
            health_results["workflow_engine"] = workflow_engine_status

            # Check Universal Agent health
            universal_agent_status = self._check_universal_agent_health()
            health_results["universal_agent"] = universal_agent_status

            # Check MCP servers health
            mcp_status = self._check_mcp_health()
            health_results["mcp_servers"] = mcp_status

            # Determine overall health
            # Only consider 'unhealthy' status as problematic, not 'disabled' services
            overall_health = "healthy"
            for component, status in health_results.items():
                component_status = status.get("status")
                if component_status == "unhealthy":
                    overall_health = "degraded"
                    break
                elif component_status not in ["healthy", "disabled"]:
                    # Handle any unexpected status as degraded
                    overall_health = "degraded"
                    break

            self.health_status = overall_health
            self.metrics["health_check"] = {
                "overall_status": overall_health,
                "components": health_results,
                "last_check": time.time(),
            }

            logger.info(f"Health check completed: {overall_health}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.health_status = "unhealthy"

    def _check_supervisor_health(self) -> Dict[str, Any]:
        """Check Supervisor component health"""
        try:
            status = self.supervisor.status()
            return {
                "status": (
                    "healthy" if status.get("status") == "running" else "unhealthy"
                ),
                "details": status,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_workflow_engine_health(self) -> Dict[str, Any]:
        """Check WorkflowEngine component health"""
        try:
            if hasattr(self.supervisor, "workflow_engine"):
                workflow_engine = self.supervisor.workflow_engine
                metrics = workflow_engine.get_workflow_metrics()

                # Check for concerning conditions
                active_workflows = metrics.get("active_workflows", 0)
                queue_size = metrics.get("queue_size", 0)

                status = "healthy"
                if active_workflows > 100:
                    status = "overloaded"
                elif queue_size > 200:
                    status = "congested"

                return {
                    "status": status,
                    "active_workflows": active_workflows,
                    "queue_size": queue_size,
                    "state": metrics.get("state", "unknown"),
                }
            else:
                return {"status": "unhealthy", "error": "WorkflowEngine not available"}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_universal_agent_health(self) -> Dict[str, Any]:
        """Check Universal Agent component health"""
        try:
            if hasattr(self.supervisor, "workflow_engine"):
                workflow_engine = self.supervisor.workflow_engine
                ua_status = workflow_engine.get_universal_agent_status()

                return {
                    "status": (
                        "healthy"
                        if ua_status.get("universal_agent_enabled")
                        else "disabled"
                    ),
                    "framework": ua_status.get("framework", "unknown"),
                    "enabled": ua_status.get("universal_agent_enabled", False),
                }
            else:
                return {"status": "unhealthy", "error": "Universal Agent not available"}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_mcp_health(self) -> Dict[str, Any]:
        """Check MCP servers health"""
        try:
            if hasattr(self.supervisor, "workflow_engine"):
                workflow_engine = self.supervisor.workflow_engine

                # Check if Universal Agent has MCP client
                if hasattr(workflow_engine, "universal_agent") and hasattr(
                    workflow_engine.universal_agent, "mcp_client"
                ):
                    mcp_client = workflow_engine.universal_agent.mcp_client

                    if mcp_client:
                        server_status = mcp_client.get_all_server_status()
                        healthy_servers = sum(
                            1
                            for status in server_status.values()
                            if status.get("status") == "connected"
                        )
                        total_servers = len(server_status)

                        overall_status = (
                            "healthy"
                            if healthy_servers == total_servers
                            else "degraded"
                        )

                        return {
                            "status": overall_status,
                            "healthy_servers": healthy_servers,
                            "total_servers": total_servers,
                            "servers": server_status,
                        }
                    else:
                        return {
                            "status": "disabled",
                            "message": "MCP client not initialized",
                        }
                else:
                    return {
                        "status": "disabled",
                        "message": "MCP integration not available",
                    }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            "overall_status": self.health_status,
            "metrics": self.metrics,
            "uptime": time.time() - getattr(self, "start_time", time.time()),
            "last_heartbeat": self.metrics.get("heartbeat", {}).get(
                "last_heartbeat", 0
            ),
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "timestamp": time.time(),
            "heartbeat": self.metrics.get("heartbeat", {}),
            "system": self.metrics.get("system", {}),
            "workflow_engine": self.metrics.get("workflow_engine", {}),
            "health_check": self.metrics.get("health_check", {}),
        }

    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.health_status == "healthy"

    def __enter__(self):
        """Context manager entry"""
        self.start_time = time.time()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
