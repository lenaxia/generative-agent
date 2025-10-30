"""
Threading Architecture Health Monitor

This module monitors the health and performance of the single event loop
threading architecture implemented in Documents 25, 26, and 27.

Created: 2025-10-13
Part of: Phase 4 - Production Deployment (Document 27)
"""

import asyncio
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ThreadingMonitor:
    """
    Monitor threading architecture health and performance.

    This monitor validates that the single event loop architecture is working
    correctly and provides metrics for production monitoring.
    """

    def __init__(self):
        """Initialize the threading monitor."""
        self._metrics = {
            "background_threads": 0,
            "scheduled_tasks": 0,
            "intent_processing_rate": 0.0,
            "handler_errors": 0,
            "last_health_check": time.time(),
            "uptime_start": time.time(),
        }

        self._intent_processing_history: list[float] = []
        self._max_history_size = 100
        self._error_count = 0
        self._last_error_time: float | None = None

    def get_threading_health(self) -> dict[str, Any]:
        """
        Get current threading health metrics.

        Returns:
            Dict containing comprehensive threading health information
        """
        current_time = time.time()
        uptime = current_time - self._metrics["uptime_start"]

        # Get current thread and task counts
        thread_count = threading.active_count()

        try:
            # Get asyncio task count (may fail if no event loop)
            loop = asyncio.get_running_loop()
            task_count = len(
                [task for task in asyncio.all_tasks(loop) if not task.done()]
            )
        except RuntimeError:
            task_count = 0

        # Calculate intent processing rate
        avg_processing_rate = self._calculate_average_processing_rate()

        return {
            "thread_count": thread_count,
            "main_thread_only": thread_count == 1,
            "scheduled_tasks": task_count,
            "uptime_seconds": uptime,
            "intent_processing_rate": avg_processing_rate,
            "error_count": self._error_count,
            "last_error_time": self._last_error_time,
            "health_status": self._determine_health_status(
                thread_count, self._error_count
            ),
            "metrics": self._metrics.copy(),
            "last_updated": current_time,
        }

    def validate_single_event_loop(self) -> bool:
        """
        Validate single event loop architecture.

        Returns:
            True if only main thread is active, False otherwise
        """
        thread_count = threading.active_count()
        is_valid = thread_count == 1

        if not is_valid:
            logger.warning(
                f"Threading validation failed: {thread_count} threads active (expected 1)"
            )
            self._record_threading_violation(thread_count)

        return is_valid

    def record_intent_processing(self, processing_time: float, intent_count: int):
        """
        Record intent processing performance metrics.

        Args:
            processing_time: Time taken to process intents (seconds)
            intent_count: Number of intents processed
        """
        if processing_time > 0:
            rate = intent_count / processing_time
            self._intent_processing_history.append(rate)

            # Maintain history size limit
            if len(self._intent_processing_history) > self._max_history_size:
                self._intent_processing_history.pop(0)

            # Update metrics
            self._metrics["intent_processing_rate"] = rate
            self._metrics["last_health_check"] = time.time()

    def record_handler_error(self, error: Exception, handler_name: str = "unknown"):
        """
        Record handler error for monitoring.

        Args:
            error: Exception that occurred
            handler_name: Name of the handler that failed
        """
        self._error_count += 1
        self._last_error_time = time.time()
        self._metrics["handler_errors"] = self._error_count

        logger.error(f"Handler error in {handler_name}: {error}")

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get performance summary for monitoring dashboards.

        Returns:
            Dict containing performance metrics summary
        """
        avg_rate = self._calculate_average_processing_rate()
        min_rate = (
            min(self._intent_processing_history)
            if self._intent_processing_history
            else 0
        )
        max_rate = (
            max(self._intent_processing_history)
            if self._intent_processing_history
            else 0
        )

        return {
            "intent_processing": {
                "average_rate": avg_rate,
                "min_rate": min_rate,
                "max_rate": max_rate,
                "sample_count": len(self._intent_processing_history),
            },
            "error_metrics": {
                "total_errors": self._error_count,
                "last_error_time": self._last_error_time,
                "error_rate": self._calculate_error_rate(),
            },
            "threading_metrics": {
                "thread_count": threading.active_count(),
                "single_thread_compliant": self.validate_single_event_loop(),
            },
        }

    def reset_metrics(self):
        """Reset all metrics (useful for testing and maintenance)."""
        self._metrics = {
            "background_threads": 0,
            "scheduled_tasks": 0,
            "intent_processing_rate": 0.0,
            "handler_errors": 0,
            "last_health_check": time.time(),
            "uptime_start": time.time(),
        }
        self._intent_processing_history.clear()
        self._error_count = 0
        self._last_error_time = None

        logger.info("Threading monitor metrics reset")

    def _calculate_average_processing_rate(self) -> float:
        """Calculate average intent processing rate."""
        if not self._intent_processing_history:
            return 0.0
        return sum(self._intent_processing_history) / len(
            self._intent_processing_history
        )

    def _calculate_error_rate(self) -> float:
        """Calculate error rate over uptime."""
        uptime = time.time() - self._metrics["uptime_start"]
        if uptime <= 0:
            return 0.0
        return self._error_count / uptime

    def _determine_health_status(self, thread_count: int, error_count: int) -> str:
        """
        Determine overall health status.

        Args:
            thread_count: Current thread count
            error_count: Total error count

        Returns:
            Health status string: "healthy", "warning", or "critical"
        """
        # Critical if multiple threads (threading violation)
        if thread_count > 1:
            return "critical"

        # Warning if recent errors
        if error_count > 0 and self._last_error_time:
            time_since_error = time.time() - self._last_error_time
            if time_since_error < 300:  # 5 minutes
                return "warning"

        # Healthy otherwise
        return "healthy"

    def _record_threading_violation(self, thread_count: int):
        """Record threading architecture violation."""
        violation_details = {
            "thread_count": thread_count,
            "expected_count": 1,
            "timestamp": time.time(),
            "active_threads": [thread.name for thread in threading.enumerate()],
        }

        logger.critical(f"Threading violation detected: {violation_details}")
        self._metrics["background_threads"] = thread_count - 1

    def get_diagnostic_info(self) -> dict[str, Any]:
        """
        Get detailed diagnostic information for troubleshooting.

        Returns:
            Dict containing detailed diagnostic information
        """
        try:
            loop = asyncio.get_running_loop()
            loop_info = {
                "loop_running": True,
                "loop_debug": loop.get_debug(),
                "loop_time": loop.time(),
            }
        except RuntimeError:
            loop_info = {
                "loop_running": False,
                "error": "No running event loop",
            }

        return {
            "threading": {
                "active_threads": [
                    {
                        "name": thread.name,
                        "daemon": thread.daemon,
                        "alive": thread.is_alive(),
                    }
                    for thread in threading.enumerate()
                ],
                "main_thread": threading.main_thread().name,
            },
            "asyncio": loop_info,
            "performance": self.get_performance_summary(),
            "health": self.get_threading_health(),
        }


# Global threading monitor instance
_threading_monitor: ThreadingMonitor | None = None


def get_threading_monitor() -> ThreadingMonitor:
    """
    Get the global threading monitor instance.

    Returns:
        ThreadingMonitor instance
    """
    global _threading_monitor
    if _threading_monitor is None:
        _threading_monitor = ThreadingMonitor()
        logger.info("Threading monitor initialized")
    return _threading_monitor


def validate_threading_architecture() -> bool:
    """
    Validate that the threading architecture is working correctly.

    Returns:
        True if threading architecture is valid, False otherwise
    """
    monitor = get_threading_monitor()
    return monitor.validate_single_event_loop()


def get_threading_health_report() -> dict[str, Any]:
    """
    Get comprehensive threading health report for monitoring.

    Returns:
        Dict containing complete threading health information
    """
    monitor = get_threading_monitor()
    return {
        "health": monitor.get_threading_health(),
        "performance": monitor.get_performance_summary(),
        "diagnostics": monitor.get_diagnostic_info(),
        "validation": {
            "single_event_loop": monitor.validate_single_event_loop(),
            "architecture_compliant": validate_threading_architecture(),
        },
    }
