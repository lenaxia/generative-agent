"""AWS Bedrock configuration for the StrandsAgent Universal Agent System.

Provides configuration management for AWS Bedrock API integration,
including credentials, model settings, and request parameters.
"""

from collections.abc import Mapping
from typing import Callable, Optional

from config.base_config import BaseConfig, ModelConfig


class BedrockModelConfig(ModelConfig):
    """Model configuration class for AWS Bedrock integration.

    Manages AWS credentials, model settings, and Bedrock-specific
    configuration parameters for foundation model access.
    """

    model: str = None
    temperature: Optional[float] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    signature_version: Optional[str] = "v4"
    defaults_mode: Optional[str] = "auto"
    beta_use_converse_api: bool = False
    # Removed callback dependencies for framework compatibility
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
    """Configuration class for AWS Bedrock API integration.

    Provides configuration management for AWS Bedrock foundation models,
    including credentials, region settings, and model parameters.
    """

    def __init__(self, **data):
        """Initialize BedrockConfig with AWS and model parameters.

        Args:
            **data: Configuration data including AWS credentials and model settings.
        """
        super().__init__(**data)
        self.llm_config = BedrockModelConfig(**data)
        self.provider_name: str = "bedrock"
