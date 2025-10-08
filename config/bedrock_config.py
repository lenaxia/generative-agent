"""AWS Bedrock configuration for the StrandsAgent Universal Agent System.

Provides configuration management for AWS Bedrock API integration,
including credentials, model settings, and request parameters.
"""

from collections.abc import Mapping
from typing import Callable, Optional

from config.base_config import BaseConfig, ModelConfig


class BedrockModelConfig(ModelConfig):
    model: str = None
    temperature: Optional[float] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    signature_version: Optional[str] = "v4"
    defaults_mode: Optional[str] = "auto"
    beta_use_converse_api: bool = False
    # Removed LangChain callback dependencies for StrandsAgent compatibility
    # callback_manager: Optional[BaseCallbackManager] = None
    # callbacks: Optional[Callbacks] = None
    credentials_profile_name: Optional[str] = None
    custom_get_token_ids: Optional[Callable[[str], list[int]]] = None
    default_headers: Optional[Mapping[str, str]] = None
    default_request_timeout: Optional[float] = None
    guardrails: Optional[Mapping[str, any]] = None
    metadata: Optional[dict[str, any]] = None
    provider_stop_reason_key_map: Optional[Mapping[str, str]] = None
    provider_stop_sequence_key_name_map: Optional[Mapping[str, str]] = None
    region_name: Optional[str] = None
    system_prompt_with_tools: str = ""
    tags: Optional[list[str]] = None


class BedrockConfig(BaseConfig):
    def __init__(self, **data):
        super().__init__(**data)
        self.llm_config = BedrockModelConfig(**data)
        self.provider_name: str = "bedrock"
