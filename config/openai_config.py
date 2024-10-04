from typing import Optional, Union, Tuple, Dict, List, Any
from .base_config import BaseConfig

class OpenAIConfig(BaseConfig):
    provider_name: str = "openai"
    frequency_penalty: float = 0.0  # No frequency penalty by default
    presence_penalty: float = 0.0  # No presence penalty by default
    logit_bias: Optional[Dict[int, int]] = None  # No logit bias by default
    seed: Optional[int] = None  # No seed specified by default
    n: int = 1  # Generate only one response by default
    logprobs: bool = False  # Do not return log probabilities by default
    top_logprobs: Optional[int] = None  # Do not return top log probabilities by default
    organization: Optional[str] = None  # No organization specified by default
    proxy: Optional[str] = None  # No proxy specified by default
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = None  # No timeout specified by default
    tiktoken_model_name: Optional[str] = None  # No tiktoken model name specified by default
    base_url: Optional[str] = None  # No base URL specified by default
    temperature: float = 0.7  # Use a temperature of 0.7 by default
    max_retries: int = 3  # Retry up to 3 times by default
    streaming: bool = False  # Do not stream responses by default
    stream_usage: bool = False  # Do not stream token usage by default

    def __init__(self, **data):
        super().__init__(provider_name="openai", **data)
        self.populate_from_env("OPENAI")
