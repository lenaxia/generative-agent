"""OpenAI API configuration for the StrandsAgent Universal Agent System.

Provides configuration management for OpenAI GPT API integration,
including API keys, model settings, and request parameters.
"""

from typing import Any, Optional, Union

from config.base_config import BaseConfig, ModelConfig


class OpenAIModelConfig(ModelConfig):
    """Model configuration class for OpenAI GPT integration.

    Manages OpenAI API settings, model parameters, and request
    configuration for GPT model access.
    """

    model: str = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    stream_options: Optional[dict] = None
    timeout: Optional[Union[float, tuple[float, float], Any]] = None
    max_retries: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    stream_usage: Optional[bool] = None

    def __init__(self, **data):
        """Initialize OpenAIModelConfig with model parameters.

        Args:
            **data: Configuration data including model and API settings.
        """
        super().__init__(**data)


class OpenAIConfig(BaseConfig):
    """Configuration class for OpenAI GPT API integration.

    Provides configuration management for OpenAI GPT models,
    including API keys, model settings, and request parameters.
    """

    def __init__(self, **data):
        """Initialize OpenAIConfig with API and model parameters.

        Args:
            **data: Configuration data including API credentials and model settings.
        """
        super().__init__(**data)
        self.llm_config = OpenAIModelConfig(**data)
        self.provider_name: str = "openai"
