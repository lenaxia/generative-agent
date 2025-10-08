"""Base configuration classes for the StrandsAgent Universal Agent System.

Provides abstract base classes and common configuration patterns
for all LLM provider configurations in the system.
"""

import os
from typing import Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    class Config:
        env_prefix = ""
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

    def populate_from_env(self, provider, name):
        env_prefix = f"{provider.upper()}_{name.upper()}_"
        for field in self.model_fields:
            env_var_name = env_prefix + field.upper()
            if env_var_name in os.environ:
                setattr(self, field, os.environ[env_var_name])


class BaseConfig(BaseModel):
    name: str
    provider_name: str = "baseconfig"
    api_key: Optional[str] = None
    llm_config_class: type[
        ModelConfig
    ] = None  # Add a new field for the model configuration class
    llm_config: ModelConfig = None

    class Config:
        env_prefix = ""
        arbitrary_types_allowed = True

    def __init__(self, **data):
        llm_config_data = {}
        for key, value in data.items():
            if key not in self.model_fields:
                llm_config_data[key] = value

        super().__init__(**{k: v for k, v in data.items() if k in self.model_fields})

        if self.llm_config_class:
            self.llm_config = self.llm_config_class(**llm_config_data)
        else:
            self.llm_config = ModelConfig(**llm_config_data)

        self.populate_from_env(self.provider_name, self.name)

    def populate_from_env(self, provider_name, name):
        """
        Populate the configuration fields from environment variables.

        The environment variables are of the form `<PROVIDER_NAME>_<NAME>_<FIELD_NAME>`, where `<PROVIDER_NAME>` is the
        provider name (e.g. "openai"), `<NAME>` is the name of the configuration (e.g. "default"), and `<FIELD_NAME>` is
        the name of the field (e.g. "api_key").

        This method is called automatically in the constructor, but can also be called manually to re-populate the
        configuration from environment variables.

        :param provider_name: The name of the provider (e.g. "openai").
        :param name: The name of the configuration (e.g. "default").
        :return: None
        """
        env_prefix = f"{provider_name.upper()}_{name.upper()}_"
        for field in self.model_fields:
            env_var_name = env_prefix + field.upper()
            if env_var_name in os.environ:
                setattr(self, field, os.environ[env_var_name])

        self.llm_config.populate_from_env(provider_name, name)
