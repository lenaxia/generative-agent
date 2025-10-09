"""Anthropic API configuration for the StrandsAgent Universal Agent System.

Provides configuration management for Anthropic Claude API integration,
including API keys, model settings, and request parameters.
"""

import os
from collections.abc import Mapping
from typing import Any, Callable, Optional

from .base_config import BaseConfig


class AnthropicConfig(BaseConfig):
    """Configuration class for Anthropic Claude API integration.

    Manages API keys, model settings, request parameters, and other
    configuration options specific to Anthropic's Claude models.
    """

    provider_name: str = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = "claude-v1.3",
        endpoint: Optional[str] = "https://api.anthropic.com/v1/completions",
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        temperature: Optional[float] = 0.7,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = 2,
        api_base: Optional[str] = None,
        # Removed callback dependencies for framework compatibility
        # callback_manager: Optional[BaseCallbackManager] = None,
        # callbacks: Optional[Callbacks] = None,
        custom_get_token_ids: Optional[Callable[[str], list[int]]] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        default_request_timeout: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        stream_usage: bool = True,
        tags: Optional[list[str]] = None,
        **base_config_kwargs,
    ):
        """Initialize AnthropicConfig with API and model parameters.

        Args:
            api_key: Anthropic API key for authentication.
            model_name: Name of the Claude model to use.
            endpoint: API endpoint URL for Anthropic services.
            top_p: Nucleus sampling parameter for response generation.
            top_k: Top-k sampling parameter for response generation.
            temperature: Sampling temperature for response randomness.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            api_base: Base URL for API requests.
            custom_get_token_ids: Custom tokenization function.
            default_headers: Default HTTP headers for requests.
            default_request_timeout: Default timeout for requests.
            metadata: Additional metadata for the configuration.
            model_kwargs: Additional model-specific parameters.
            stream_usage: Whether to enable streaming usage tracking.
            tags: Tags for request categorization.
            **base_config_kwargs: Additional base configuration parameters.
        """
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            endpoint=endpoint,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_retries=max_retries,
            api_base=api_base,
            # rate_limiter=rate_limiter,  # Undefined variable removed
            stream_usage=stream_usage,
            streaming=False,
            **base_config_kwargs,
        )
        self.timeout = os.environ.get("ANTHROPIC_TIMEOUT", timeout)
        # self.cache = os.environ.get("ANTHROPIC_CACHE", cache)  # Undefined variable
        # Removed callback dependencies
        # self.callback_manager = os.environ.get("ANTHROPIC_CALLBACK_MANAGER", callback_manager)
        # self.callbacks = os.environ.get("ANTHROPIC_CALLBACKS", callbacks)
        self.custom_get_token_ids = os.environ.get(
            "ANTHROPIC_CUSTOM_GET_TOKEN_IDS", custom_get_token_ids
        )
        self.default_headers = os.environ.get(
            "ANTHROPIC_DEFAULT_HEADERS", default_headers
        )
        self.default_request_timeout = os.environ.get(
            "ANTHROPIC_DEFAULT_REQUEST_TIMEOUT", default_request_timeout
        )
        # self.disable_streaming = os.environ.get("ANTHROPIC_DISABLE_STREAMING", disable_streaming)  # Undefined variable
        self.metadata = os.environ.get("ANTHROPIC_METADATA", metadata)
        self.model_kwargs = os.environ.get("ANTHROPIC_MODEL_KWARGS", model_kwargs)
        self.tags = os.environ.get("ANTHROPIC_TAGS", tags)
