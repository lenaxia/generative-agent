import os
from typing import Optional, Union, Tuple, Dict, List, Any, Mapping, Callable
from langchain_core.rate_limits import BaseRateLimiter
from langchain_core.cache import BaseCache
from langchain_core.callbacks.manager import BaseCallbackManager, Callbacks

class OpenAIConfig(BaseConfig):
    provider_name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = "text-davinci-003",
        endpoint: Optional[str] = "https://api.openai.com/v1/chat/completions",
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        temperature: Optional[float] = 0.7,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, int]] = None,
        seed: Optional[int] = None,
        n: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_retries: Optional[int] = 3,
        organization: Optional[str] = None,
        proxy: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float], Any]] = None,
        tiktoken_model_name: Optional[str] = None,
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
            **base_config_kwargs,
        )
        self.frequency_penalty = os.environ.get("OPENAI_FREQUENCY_PENALTY", frequency_penalty)
        self.presence_penalty = os.environ.get("OPENAI_PRESENCE_PENALTY", presence_penalty)
        self.logit_bias = os.environ.get("OPENAI_LOGIT_BIAS", logit_bias)
        self.seed = os.environ.get("OPENAI_SEED", seed)
        self.n = os.environ.get("OPENAI_N", n)
        self.logprobs = os.environ.get("OPENAI_LOGPROBS", logprobs)
        self.top_logprobs = os.environ.get("OPENAI_TOP_LOGPROBS", top_logprobs)
        self.organization = os.environ.get("OPENAI_ORGANIZATION", organization)
        self.proxy = os.environ.get("OPENAI_PROXY", proxy)
        self.request_timeout = os.environ.get("OPENAI_REQUEST_TIMEOUT", request_timeout)
        self.tiktoken_model_name = os.environ.get("OPENAI_TIKTOKEN_MODEL_NAME", tiktoken_model_name)

