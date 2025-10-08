"""Configuration manager module for supervisor settings and parameters.

This module provides configuration management functionality for the supervisor
system, handling settings, parameters, and configuration validation.
"""

import logging
from typing import Optional

import yaml
from pydantic import BaseModel, ValidationError

from supervisor.supervisor_config import SupervisorConfig

logger = logging.getLogger(__name__)


class ConfigManager(BaseModel):
    config_file: Optional[str] = None

    def __init__(self, config_file: str):
        super().__init__(config_file=config_file)

    def load_config(self):
        try:
            with open(self.config_file) as f:
                config_data = yaml.safe_load(f)
            config = SupervisorConfig(**config_data)
            return config
        except ValidationError as e:
            logger.error(f"Error validating configuration: {e}")
            raise e
        except Exception as e:
            logger.error(
                f"Error loading configuration from file '{self.config_file}': {e}"
            )
            raise e
