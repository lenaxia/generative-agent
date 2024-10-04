from typing import Optional, Dict, List, Mapping, Callable, Any
from langchain_core.callbacks.manager import BaseCallbackManager, Callbacks
from .base_config import BaseConfig
from pydantic import validator

class BedrockConfig(BaseConfig):
    model_name: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    signature_version: Optional[str] = "v4"
    defaults_mode: Optional[str] = "auto"
    beta_use_converse_api: bool = False
    callback_manager: Optional[BaseCallbackManager] = None
    callbacks: Optional[Callbacks] = None
    credentials_profile_name: Optional[str] = None
    custom_get_token_ids: Optional[Callable[[str], List[int]]] = None
    default_headers: Optional[Mapping[str, str]] = None
    default_request_timeout: Optional[float] = None
    guardrails: Optional[Mapping[str, any]] = None
    metadata: Optional[Dict[str, any]] = None
    provider: Optional[str] = None
    provider_stop_reason_key_map: Optional[Mapping[str, str]] = None
    provider_stop_sequence_key_name_map: Optional[Mapping[str, str]] = None
    region_name: Optional[str] = None
    system_prompt_with_tools: str = ''
    tags: Optional[List[str]] = None

    def __init__(self, **data):
        super().__init__(provider_name="bedrock", **data)
        self.populate_from_env("BEDROCK")
