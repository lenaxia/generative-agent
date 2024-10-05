from typing import Optional, Union, Tuple, Dict, List, Any
from .base_config import BaseConfig

class OpenAIConfig(BaseConfig):
    provider_name: str = "openai"
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[Dict[int, int]] = None
    seed: Optional[int] = None
    n: int = 1
    logprobs: bool = False  # Add this line
    top_logprobs: Optional[int] = None  # Add this line
    organization: Optional[str] = None
    proxy: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = None
    tiktoken_model_name: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_retries: int = 3
    streaming: bool = False
    stream_usage: bool = False
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    #request_kwargs: Optional[Dict] = None  # Add this line

    def __init__(self, **data):
        super().__init__(provider_name="openai", **data)
        self.populate_from_env("OPENAI")
