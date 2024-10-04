import os
from typing import Optional, Union, Tuple, Dict, List, Any, Mapping, Callable
from langchain_core.rate_limits import BaseRateLimiter
from langchain_core.cache import BaseCache
from langchain_core.callbacks.manager import BaseCallbackManager, Callbacks

class ChatBedrockConfig(BaseConfig):
    provider_name = "aws"

    def __init__(
        self,
        model_id: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        beta_use_converse_api: bool = False,
        cache: Optional[BaseCache] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        callbacks: Optional[Callbacks] = None,
        credentials_profile_name: Optional[str] = None,
        custom_get_token_ids: Optional[Callable[[str], List[int]]] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        default_request_timeout: Optional[float] = None,
        disable_streaming: bool | Literal['tool_calling'] = False,
        guardrails: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        provider_stop_reason_key_map: Optional[Mapping[str, str]] = None,
        provider_stop_sequence_key_name_map: Optional[Mapping[str, str]] = None,
        rate_limiter: Optional[BaseRateLimiter] = None,
        region_name: Optional[str] = None,
        streaming: bool = False,
        system_prompt_with_tools: str = '',
        tags: Optional[List[str]] = None,
        verbose: Optional[bool] = None,
        **base_config_kwargs,
    ):
        super().__init__(
            api_key=aws_access_key_id,
            model_name=model_id,
            endpoint=None,
            top_p=None,
            top_k=None,
            temperature=None,
            max_retries=None,
            api_base=None,
            rate_limiter=rate_limiter,
            stop_sequences=None,
            stream_usage=None,
            streaming=streaming,
            verbose=verbose,
            **base_config_kwargs,
        )
        self.model_id = model_id
        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", aws_access_key_id)
        self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)
        self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN", aws_session_token)
        self.beta_use_converse_api = beta_use_converse_api
        self.cache = cache
        self.callback_manager = callback_manager
        self.callbacks = callbacks
        self.credentials_profile_name = credentials_profile_name
        self.custom_get_token_ids = custom_get_token_ids
        self.default_headers = default_headers
        self.default_request_timeout = default_request_timeout
        self.disable_streaming = disable_streaming
        self.guardrails = guardrails
        self.metadata = metadata
        self.model_kwargs = model_kwargs
        self.provider = provider
        self.provider_stop_reason_key_map = provider_stop_reason_key_map
        self.provider_stop_sequence_key_name_map = provider_stop_sequence_key_name_map
        self.rate_limiter = rate_limiter
        self.region_name = region_name
        self.streaming = streaming
        self.system_prompt_with_tools = system_prompt_with_tools
        self.tags = tags
        self.verbose = verbose

