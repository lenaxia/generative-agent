"""Metrics management module for collecting and reporting system metrics.

This module provides functionality for collecting, aggregating, and reporting
performance metrics, statistics, and operational data from the supervisor
and its managed workflows.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetricsManager(BaseModel):
    # TODO: Enable configuration of the metrics manager
    # config: BaseModel = Field(..., description="The configuration object")
    metrics: dict[str, dict] = Field(
        default_factory=dict, description="Dictionary to store metrics"
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()

    def get_metrics(self, request_id: Optional[str] = None) -> dict:
        try:
            if request_id:
                return self.metrics.get(request_id, {})
            return self.metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    def update_metrics(self, request_id: str, updates: dict):
        try:
            if request_id not in self.metrics:
                self.metrics[request_id] = {}
            self.metrics[request_id].update(updates)
        except Exception as e:
            logger.error(f"Error updating metrics for request '{request_id}': {e}")

    def delta_metrics(self, request_id: str, increments: dict):
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
