import logging
from typing import Dict, Optional
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MetricsManager(BaseModel):
    config: BaseModel = Field(..., description="The configuration object")
    metrics: Dict[str, Dict] = Field(default_factory=dict, description="Dictionary to store metrics")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config):
        super().__init__(config=config)

    def get_metrics(self, request_id: Optional[str] = None) -> Dict:
        try:
            if request_id:
                return self.metrics.get(request_id, {})
            return self.metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    def update_metrics(self, request_id: str, updates: Dict):
        try:
            if request_id not in self.metrics:
                self.metrics[request_id] = {}
            self.metrics[request_id].update(updates)
        except Exception as e:
            logger.error(f"Error updating metrics for request '{request_id}': {e}")

    def persist_metrics(self, request_id: str, request_data: Dict):
        try:
            # Implement logic to persist the request data and metrics
            # to a persistent storage (e.g., database, file system)
            storage_path = Path("storage") / f"{request_id}.yaml"
            storage_path.parent.mkdir(exist_ok=True)
            with open(storage_path, "w") as f:
                yaml.safe_dump(request_data, f)
            logger.info(f"Persisted metrics for request '{request_id}' to '{storage_path}'.")
        except Exception as e:
            logger.error(f"Error persisting metrics for request '{request_id}': {e}")

    def load_metrics(self, request_id: str) -> Dict:
        try:
            storage_path = Path("storage") / f"{request_id}.yaml"
            if not storage_path.exists():
                logger.error(f"Metrics for request '{request_id}' not found in persistent storage.")
                return {}

            with open(storage_path, "r") as f:
                request_data = yaml.safe_load(f)

            logger.info(f"Loaded metrics for request '{request_id}' from '{storage_path}'.")
            return request_data
        except Exception as e:
            logger.error(f"Error loading metrics for request '{request_id}': {e}")
            return {}
