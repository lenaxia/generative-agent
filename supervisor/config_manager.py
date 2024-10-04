import logging
import yaml
from pydantic import ValidationError
from config import SupervisorConfig

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file

    def load_config(self):
        try:
            with open(self.config_file, "r") as f:
                config_data = yaml.safe_load(f)
            config = SupervisorConfig(**config_data)
            return config
        except ValidationError as e:
            logger.error(f"Error validating configuration: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error loading configuration from file '{self.config_file}': {e}")
            raise e
