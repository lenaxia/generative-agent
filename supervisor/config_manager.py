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
    """Configuration management for the supervisor system.

    Handles loading, validation, and management of supervisor
    configuration files and settings.
    """

    config_file: Optional[str] = None
    raw_config_data: Optional[dict] = None

    def __init__(self, config_file: str):
        """Initialize ConfigManager with configuration file path.

        Args:
            config_file: Path to the configuration file to manage.
        """
        super().__init__(config_file=config_file)

    def load_config(self):
        """Load and validate configuration from the specified file.

        Reads the YAML configuration file, parses it, and validates it
        against the SupervisorConfig schema.

        Returns:
            SupervisorConfig: The validated configuration object.

        Raises:
            ValidationError: If the configuration data is invalid.
            Exception: If there's an error reading or parsing the file.
        """
        try:
            with open(self.config_file) as f:
                config_data = yaml.safe_load(f)
            # Store raw config data for access to non-Pydantic fields
            self.raw_config_data = config_data
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
