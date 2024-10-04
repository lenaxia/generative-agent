import os
from typing import Optional, Union, Tuple, Dict, List, Any, Mapping, Callable
from langchain_core.rate_limits import BaseRateLimiter
from langchain_core.cache import BaseCache
from langchain_core.callbacks.manager import BaseCallbackManager, Callbacks

class BaseConfig:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        api_base: Optional[str] = None,
        rate_limiter: Optional[BaseRateLimiter] = None,
        stop_sequences: Optional[List[str]] = None,
        stream_usage: Optional[bool] = None,
        streaming: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ):
        self.api_key = os.environ.get(f"{self.provider_name.upper()}_API_KEY", api_key)
        self.model_name = os.environ.get(f"{self.provider_name.upper()}_MODEL_NAME", model_name)
        self.endpoint = os.environ.get(f"{self.provider_name.upper()}_ENDPOINT", endpoint)
        self.top_p = os.environ.get(f"{self.provider_name.upper()}_TOP_P", top_p)
        self.top_k = os.environ.get(f"{self.provider_name.upper()}_TOP_K", top_k)
        self.temperature = os.environ.get(f"{self.provider_name.upper()}_TEMPERATURE", temperature)
        self.max_retries = os.environ.get(f"{self.provider_name.upper()}_MAX_RETRIES", max_retries)
        self.api_base = os.environ.get(f"{self.provider_name.upper()}_API_BASE", api_base)
        self.rate_limiter = os.environ.get(f"{self.provider_name.upper()}_RATE_LIMITER", rate_limiter)
        self.stop_sequences = os.environ.get(f"{self.provider_name.upper()}_STOP_SEQUENCES", stop_sequences)
        self.stream_usage = os.environ.get(f"{self.provider_name.upper()}_STREAM_USAGE", stream_usage)
        self.streaming = os.environ.get(f"{self.provider_name.upper()}_STREAMING", streaming)
        self.verbose = os.environ.get(f"{self.provider_name.upper()}_VERBOSE", verbose)

