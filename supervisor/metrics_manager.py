"""Metrics management module for collecting and reporting system metrics.

This module provides functionality for collecting, aggregating, and reporting
performance metrics, statistics, and operational data from the supervisor
and its managed workflows.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetricsManager(BaseModel):
    """Metrics collection and management for the supervisor system.

    Provides centralized metrics storage, retrieval, and management
    capabilities for monitoring system performance and behavior.
    """

    # TODO: Enable configuration of the metrics manager
    # config: BaseModel = Field(..., description="The configuration object")
    metrics: dict[str, dict] = Field(
        default_factory=dict, description="Dictionary to store metrics"
    )

    class Config:
        """Pydantic configuration for MetricsManager.

        Allows arbitrary types to be used in the metrics system.
        """

        arbitrary_types_allowed = True

    def __init__(self):
        """Initialize MetricsManager with empty metrics storage.

        Sets up the metrics collection system with default configuration
        and empty metrics dictionary.
        """
        super().__init__()

    def get_metrics(self, request_id: Optional[str] = None) -> dict:
        """Retrieve metrics for a specific request or all metrics.

        Gets stored metrics data either for a specific request ID or
        returns all collected metrics if no request ID is specified.

        Args:
            request_id: Optional request identifier to get specific metrics.
                       If None, returns all metrics.

        Returns:
            Dictionary containing the requested metrics data, or empty dict on error.
        """
        try:
            if request_id:
                return self.metrics.get(request_id, {})
            return self.metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    def update_metrics(self, request_id: str, updates: dict):
        """Update metrics for a specific request with new values.

        Merges the provided updates dictionary with existing metrics
        for the specified request, creating a new entry if needed.

        Args:
            request_id: The request identifier to update metrics for.
            updates: Dictionary of metric updates to apply.
        """
        try:
            if request_id not in self.metrics:
                self.metrics[request_id] = {}
            self.metrics[request_id].update(updates)
        except Exception as e:
            logger.error(f"Error updating metrics for request '{request_id}': {e}")

    def delta_metrics(self, request_id: str, increments: dict):
        """Increment metrics for a specific request by delta values.

        Adds the provided increment values to existing metrics for the
        specified request, initializing metrics to 0 if they don't exist.

        Args:
            request_id: The request identifier to increment metrics for.
            increments: Dictionary of metric increments to apply.
        """
        try:
            if request_id not in self.metrics:
                self.metrics[request_id] = {}
            for key, value in increments.items():
                self.metrics[request_id][key] = (
                    self.metrics[request_id].get(key, 0) + value
                )
        except Exception as e:
            logger.error(f"Error incrementing metrics for request '{request_id}': {e}")

    def persist_metrics(self, request_id: str, request_data: dict):
        """Persist metrics and request data to storage.

        Saves the provided request data and metrics to a YAML file
        in the logs/storage directory for later retrieval and analysis.

        Args:
            request_id: The request identifier for the data to persist.
            request_data: Dictionary containing request data and metrics to save.
        """
        try:
            # TODO: Implement logic to persist the request data and metrics
            #       to a persistent storage (e.g., database, file system)
            storage_path = Path("logs/storage") / f"{request_id}.yaml"
            storage_path.parent.mkdir(exist_ok=True)

            with open(storage_path, "w") as f:
                yaml.safe_dump(request_data, f)
            logger.info(
                f"Persisted metrics for request '{request_id}' to '{storage_path}'."
            )
        except Exception as e:
            logger.error(f"Error persisting metrics for request '{request_id}': {e}")

    def load_metrics(self, request_id: str) -> dict:
        """Load persisted metrics and request data from storage.

        Retrieves previously saved request data and metrics from the
        storage directory for the specified request ID.

        Args:
            request_id: The request identifier for the data to load.

        Returns:
            Dictionary containing the loaded request data and metrics,
            or empty dict if not found or on error.
        """
        try:
            storage_path = Path("storage") / f"{request_id}.yaml"
            if not storage_path.exists():
                logger.error(
                    f"Metrics for request '{request_id}' not found in persistent storage."
                )
                return {}

            with open(storage_path) as f:
                request_data = yaml.safe_load(f)

            logger.info(
                f"Loaded metrics for request '{request_id}' from '{storage_path}'."
            )
            return request_data
        except Exception as e:
            logger.error(f"Error loading metrics for request '{request_id}': {e}")
            return {}
