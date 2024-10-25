from typing import Optional, Union, Tuple, Any, Dict
from config.base_config import BaseConfig, ModelConfig

class OpenAIModelConfig(ModelConfig):
    model: str = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    stream_options: Optional[Dict] = None
    timeout: Optional[Union[float, Tuple[float, float], Any]] = None
    max_retries: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    stream_usage: Optional[bool] = None

    def __init__(self, **data):
        super().__init__(**data)

class OpenAIConfig(BaseConfig):
    def __init__(self, **data):
        super().__init__(**data)
        self.llm_config = OpenAIModelConfig(**data)
        self.provider_name: str = "openai"