"""
Intent Processing Metrics and Monitoring Dashboard

This module provides comprehensive metrics collection and monitoring for the
intent processing system in the LLM-safe architecture.

Created: 2025-10-13
Part of: Technical Debt Cleanup - Intent Processing Metrics
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


class IntentMetricsCollector:
    """
    Collects and aggregates metrics for intent processing.

    Tracks performance, errors, and usage patterns for monitoring and optimization.
    """

    def __init__(self, max_history_size: int = 1000):
        """Initialize the metrics collector."""
        self.max_history_size = max_history_size

        # Core metrics
        self._total_intents_processed = 0
        self._total_intents_failed = 0
        self._total_processing_time = 0.0

        # Intent type metrics
        self._intent_type_counts = defaultdict(int)
        self._intent_type_failures = defaultdict(int)
        self._intent_type_processing_times = defaultdict(list)

        # Performance history
        self._processing_times = deque(maxlen=max_history_size)
        self._throughput_history = deque(maxlen=max_history_size)

        # Error tracking
        self._error_history = deque(maxlen=max_history_size)
        self._error_types = defaultdict(int)

        # Security metrics
        self._security_violations = 0
        self._pii_detections = 0

        # Role-specific metrics
        self._role_intent_counts = defaultdict(int)
        self._role_processing_times = defaultdict(list)

        # Start time for uptime calculation
        self._start_time = time.time()

    def record_intent_processed(
        self,
        intent_type: str,
        processing_time: float,
        role_name: str | None = None,
        success: bool = True,
    ):
        """Record a processed intent with metrics."""
        self._total_intents_processed += 1
        self._total_processing_time += processing_time

        # Record by intent type
        self._intent_type_counts[intent_type] += 1
        self._intent_type_processing_times[intent_type].append(processing_time)

        # Maintain processing time history
        if len(self._intent_type_processing_times[intent_type]) > 100:
            self._intent_type_processing_times[intent_type].pop(0)

        # Record performance metrics
        self._processing_times.append(processing_time)

        # Calculate current throughput (intents per second over last minute)
        current_time = time.time()
        recent_intents = sum(1 for t in self._processing_times if current_time - t < 60)
        throughput = recent_intents / 60.0
        self._throughput_history.append(throughput)

        # Record role-specific metrics
        if role_name:
            self._role_intent_counts[role_name] += 1
            self._role_processing_times[role_name].append(processing_time)
            if len(self._role_processing_times[role_name]) > 100:
                self._role_processing_times[role_name].pop(0)

        # Record failures
        if not success:
            self._total_intents_failed += 1
            self._intent_type_failures[intent_type] += 1

    def record_intent_error(
        self, error: Exception, intent_type: str, role_name: str | None = None
    ):
        """Record an intent processing error."""
        error_entry = {
            "timestamp": time.time(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "intent_type": intent_type,
            "role_name": role_name,
        }

        self._error_history.append(error_entry)
        self._error_types[error.__class__.__name__] += 1

        logger.error(f"Intent processing error recorded: {error_entry}")

    def record_security_event(self, event_type: str, details: dict[str, Any]):
        """Record a security-related event."""
        if event_type == "security_violation":
            self._security_violations += 1
        elif event_type == "pii_detection":
            self._pii_detections += 1

        logger.warning(f"Security event recorded: {event_type} - {details}")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        current_time = time.time()
        uptime = current_time - self._start_time

        # Calculate averages
        avg_processing_time = self._total_processing_time / max(
            1, self._total_intents_processed
        )

        # Calculate current throughput
        current_throughput = (
            self._throughput_history[-1] if self._throughput_history else 0.0
        )

        # Calculate error rate
        error_rate = self._total_intents_failed / max(1, self._total_intents_processed)

        return {
            "uptime_seconds": uptime,
            "total_intents_processed": self._total_intents_processed,
            "total_intents_failed": self._total_intents_failed,
            "average_processing_time": avg_processing_time,
            "current_throughput": current_throughput,
            "error_rate": error_rate,
            "security_violations": self._security_violations,
            "pii_detections": self._pii_detections,
            "intent_types_processed": len(self._intent_type_counts),
            "roles_active": len(self._role_intent_counts),
            "last_updated": current_time,
        }

    def get_detailed_metrics(self) -> dict[str, Any]:
        """Get detailed metrics including breakdowns by type and role."""
        current_metrics = self.get_current_metrics()

        # Intent type breakdown
        intent_type_metrics = {}
        for intent_type, count in self._intent_type_counts.items():
            processing_times = self._intent_type_processing_times[intent_type]
            avg_time = (
                sum(processing_times) / len(processing_times) if processing_times else 0
            )

            intent_type_metrics[intent_type] = {
                "count": count,
                "failures": self._intent_type_failures[intent_type],
                "average_processing_time": avg_time,
                "success_rate": (count - self._intent_type_failures[intent_type])
                / max(1, count),
            }

        # Role breakdown
        role_metrics = {}
        for role_name, count in self._role_intent_counts.items():
            processing_times = self._role_processing_times[role_name]
            avg_time = (
                sum(processing_times) / len(processing_times) if processing_times else 0
            )

            role_metrics[role_name] = {
                "intent_count": count,
                "average_processing_time": avg_time,
            }

        # Performance trends
        performance_trends = {
            "processing_time_trend": list(self._processing_times)[
                -50:
            ],  # Last 50 samples
            "throughput_trend": list(self._throughput_history)[-50:],  # Last 50 samples
        }

        # Error analysis
        error_analysis = {
            "recent_errors": list(self._error_history)[-10:],  # Last 10 errors
            "error_types": dict(self._error_types),
            "total_errors": len(self._error_history),
        }

        return {
            "current": current_metrics,
            "intent_types": intent_type_metrics,
            "roles": role_metrics,
            "performance_trends": performance_trends,
            "errors": error_analysis,
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get health status based on metrics."""
        current_metrics = self.get_current_metrics()

        # Determine health status
        health_status = "healthy"
        health_issues = []

        # Check error rate
        if current_metrics["error_rate"] > 0.1:  # >10% error rate
            health_status = "critical"
            health_issues.append(
                f"High error rate: {current_metrics['error_rate']:.1%}"
            )
        elif current_metrics["error_rate"] > 0.05:  # >5% error rate
            health_status = "warning"
            health_issues.append(
                f"Elevated error rate: {current_metrics['error_rate']:.1%}"
            )

        # Check processing time
        if current_metrics["average_processing_time"] > 1.0:  # >1 second average
            health_status = "warning" if health_status == "healthy" else health_status
            health_issues.append(
                f"Slow processing: {current_metrics['average_processing_time']:.3f}s avg"
            )

        # Check throughput
        if current_metrics["current_throughput"] < 1.0:  # <1 intent per second
            health_status = "warning" if health_status == "healthy" else health_status
            health_issues.append(
                f"Low throughput: {current_metrics['current_throughput']:.1f} intents/s"
            )

        # Check security violations
        if current_metrics["security_violations"] > 0:
            health_status = "warning" if health_status == "healthy" else health_status
            health_issues.append(
                f"Security violations: {current_metrics['security_violations']}"
            )

        return {
            "status": health_status,
            "issues": health_issues,
            "metrics_summary": current_metrics,
            "last_check": time.time(),
        }

    def reset_metrics(self):
        """Reset all metrics (useful for testing and maintenance)."""
        self._total_intents_processed = 0
        self._total_intents_failed = 0
        self._total_processing_time = 0.0

        self._intent_type_counts.clear()
        self._intent_type_failures.clear()
        self._intent_type_processing_times.clear()

        self._processing_times.clear()
        self._throughput_history.clear()
        self._error_history.clear()
        self._error_types.clear()

        self._security_violations = 0
        self._pii_detections = 0

        self._role_intent_counts.clear()
        self._role_processing_times.clear()

        self._start_time = time.time()

        logger.info("Intent metrics reset")


class IntentProcessingDashboard:
    """
    Dashboard for monitoring intent processing metrics.

    Provides formatted output and monitoring capabilities for intent processing.
    """

    def __init__(self, metrics_collector: IntentMetricsCollector):
        """Initialize the dashboard."""
        self.metrics_collector = metrics_collector

    def get_dashboard_summary(self) -> str:
        """Get formatted dashboard summary."""
        metrics = self.metrics_collector.get_current_metrics()
        health = self.metrics_collector.get_health_status()

        summary = f"""
=== Intent Processing Dashboard ===

Health Status: {health['status'].upper()}
{f"Issues: {', '.join(health['issues'])}" if health['issues'] else "No issues detected"}

Performance Metrics:
  • Total Intents Processed: {metrics['total_intents_processed']:,}
  • Average Processing Time: {metrics['average_processing_time']:.3f}s
  • Current Throughput: {metrics['current_throughput']:.1f} intents/second
  • Error Rate: {metrics['error_rate']:.1%}

Security Metrics:
  • Security Violations: {metrics['security_violations']}
  • PII Detections: {metrics['pii_detections']}

System Status:
  • Uptime: {metrics['uptime_seconds'] / 3600:.1f} hours
  • Intent Types Active: {metrics['intent_types_processed']}
  • Roles Active: {metrics['roles_active']}

Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics['last_updated']))}
"""
        return summary

    def get_detailed_report(self) -> str:
        """Get detailed metrics report."""
        detailed = self.metrics_collector.get_detailed_metrics()

        report = self.get_dashboard_summary()

        # Add intent type breakdown
        report += "\n=== Intent Type Breakdown ===\n"
        for intent_type, metrics in detailed["intent_types"].items():
            report += f"  • {intent_type}: {metrics['count']} processed, "
            report += f"{metrics['success_rate']:.1%} success rate, "
            report += f"{metrics['average_processing_time']:.3f}s avg\n"

        # Add role breakdown
        report += "\n=== Role Breakdown ===\n"
        for role_name, metrics in detailed["roles"].items():
            report += f"  • {role_name}: {metrics['intent_count']} intents, "
            report += f"{metrics['average_processing_time']:.3f}s avg\n"

        # Add recent errors
        if detailed["errors"]["recent_errors"]:
            report += "\n=== Recent Errors ===\n"
            for error in detailed["errors"]["recent_errors"][-5:]:  # Last 5 errors
                timestamp = time.strftime(
                    "%H:%M:%S", time.localtime(error["timestamp"])
                )
                report += f"  • {timestamp}: {error['error_type']} in {error['intent_type']}\n"

        return report

    def export_metrics_json(self) -> dict[str, Any]:
        """Export all metrics as JSON for external monitoring systems."""
        return {
            "dashboard_export": {
                "timestamp": time.time(),
                "metrics": self.metrics_collector.get_detailed_metrics(),
                "health": self.metrics_collector.get_health_status(),
            }
        }


# Global metrics collector instance
_metrics_collector: IntentMetricsCollector | None = None
_dashboard: IntentProcessingDashboard | None = None


def get_intent_metrics_collector() -> IntentMetricsCollector:
    """
    Get the global intent metrics collector instance.

    Returns:
        IntentMetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = IntentMetricsCollector()
        logger.info("Intent metrics collector initialized")
    return _metrics_collector


def get_intent_dashboard() -> IntentProcessingDashboard:
    """
    Get the global intent processing dashboard instance.

    Returns:
        IntentProcessingDashboard instance
    """
    global _dashboard
    if _dashboard is None:
        collector = get_intent_metrics_collector()
        _dashboard = IntentProcessingDashboard(collector)
        logger.info("Intent processing dashboard initialized")
    return _dashboard


def record_intent_processing_metrics(
    intent_type: str,
    processing_time: float,
    role_name: str | None = None,
    success: bool = True,
):
    """
    Record intent processing metrics.

    Args:
        intent_type: Type of intent processed
        processing_time: Time taken to process the intent
        role_name: Name of the role that processed the intent
        success: Whether processing was successful
    """
    collector = get_intent_metrics_collector()
    collector.record_intent_processed(intent_type, processing_time, role_name, success)


def record_intent_error(
    error: Exception, intent_type: str, role_name: str | None = None
):
    """
    Record an intent processing error.

    Args:
        error: Exception that occurred
        intent_type: Type of intent that failed
        role_name: Name of the role where error occurred
    """
    collector = get_intent_metrics_collector()
    collector.record_intent_error(error, intent_type, role_name)


def get_intent_processing_health() -> dict[str, Any]:
    """
    Get intent processing health status.

    Returns:
        Dict containing health status and metrics
    """
    collector = get_intent_metrics_collector()
    return collector.get_health_status()


def get_intent_dashboard_summary() -> str:
    """
    Get formatted dashboard summary.

    Returns:
        Formatted string with dashboard information
    """
    dashboard = get_intent_dashboard()
    return dashboard.get_dashboard_summary()


def export_intent_metrics() -> dict[str, Any]:
    """
    Export intent metrics for external monitoring.

    Returns:
        Dict containing all metrics data
    """
    dashboard = get_intent_dashboard()
    return dashboard.export_metrics_json()
