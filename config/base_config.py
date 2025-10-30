"""Base configuration classes for the StrandsAgent Universal Agent System.

Provides abstract base classes and common configuration patterns
for all LLM provider configurations in the system.
"""

import os
from typing import Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Base model configuration class for LLM providers.

    Provides common configuration patterns and environment variable
    population for model-specific settings.
    """

    class Config:
        """Pydantic configuration for ModelConfig.

        Allows arbitrary types and configures environment variable prefix.
        """

        env_prefix = ""
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize ModelConfig with provided data.

        Args:
            **data: Configuration data as keyword arguments.
        """
        super().__init__(**data)

    def populate_from_env(self, provider, name):
        """Populate configuration fields from environment variables.

        Reads environment variables with the pattern PROVIDER_NAME_FIELD
        and sets the corresponding model fields if the variables exist.

        Args:
            provider: The provider name (e.g., 'openai', 'anthropic').
            name: The configuration name (e.g., 'default', 'strong').
        """
        env_prefix = f"{provider.upper()}_{name.upper()}_"
        for field in self.model_fields:
            env_var_name = env_prefix + field.upper()
            if env_var_name in os.environ:
                setattr(self, field, os.environ[env_var_name])


class BaseConfig(BaseModel):
    """Abstract base configuration class for all LLM providers.

    Provides common configuration fields and environment variable
    population functionality for all provider-specific configurations.
    """

    name: str
    provider_name: str = "baseconfig"
    api_key: str | None = None
    llm_config_class: type[
        ModelConfig
    ] = None  # Add a new field for the model configuration class
    llm_config: ModelConfig = None

    class Config:
        """Pydantic configuration for BaseConfig.

        Allows arbitrary types and configures environment variable prefix.
        """

        env_prefix = ""
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize BaseConfig with provider-specific configuration.

        Args:
            **data: Configuration data including provider-specific parameters.
        """
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
        """Populate the configuration fields from environment variables.

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
