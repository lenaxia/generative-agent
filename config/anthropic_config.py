import os
from typing import Optional, Union, Tuple, Dict, List, Any, Mapping, Callable
from langchain_core.callbacks.manager import BaseCallbackManager, Callbacks
from .base_config import BaseConfig

class AnthropicConfig(BaseConfig):
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
        callback_manager: Optional[BaseCallbackManager] = None,
        callbacks: Optional[Callbacks] = None,
        custom_get_token_ids: Optional[Callable[[str], List[int]]] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        default_request_timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        stream_usage: bool = True,
        tags: Optional[List[str]] = None,
        **base_config_kwargs,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            endpoint=endpoint,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_retries=max_retries,
            api_base=api_base,
            rate_limiter=rate_limiter,
            stream_usage=stream_usage,
            streaming=False,
            **base_config_kwargs,
        )
        self.timeout = os.environ.get("ANTHROPIC_TIMEOUT", timeout)
        self.cache = os.environ.get("ANTHROPIC_CACHE", cache)
        self.callback_manager = os.environ.get("ANTHROPIC_CALLBACK_MANAGER", callback_manager)
        self.callbacks = os.environ.get("ANTHROPIC_CALLBACKS", callbacks)
        self.custom_get_token_ids = os.environ.get("ANTHROPIC_CUSTOM_GET_TOKEN_IDS", custom_get_token_ids)
        self.default_headers = os.environ.get("ANTHROPIC_DEFAULT_HEADERS", default_headers)
        self.default_request_timeout = os.environ.get("ANTHROPIC_DEFAULT_REQUEST_TIMEOUT", default_request_timeout)
        self.disable_streaming = os.environ.get("ANTHROPIC_DISABLE_STREAMING", disable_streaming)
        self.metadata = os.environ.get("ANTHROPIC_METADATA", metadata)
        self.model_kwargs = os.environ.get("ANTHROPIC_MODEL_KWARGS", model_kwargs)
        self.tags = os.environ.get("ANTHROPIC_TAGS", tags)
