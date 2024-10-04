from typing import Optional, List
from pydantic import BaseModel, Field, validator
import os

class BaseConfig(BaseModel):
    name: str
    provider_name: str
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    endpoint: Optional[str] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_retries: Optional[int] = None
    api_base: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    stream_usage: Optional[bool] = None
    streaming: Optional[bool] = None
    verbose: Optional[bool] = None

    @validator('api_key', pre=True, always=True)
    def set_api_key(cls, api_key, values):
        provider_name = values.get('provider_name')
        if provider_name:
            env_var_name = f"{provider_name.upper()}_API_KEY"
            api_key = api_key or os.environ.get(env_var_name)
        return api_key

    class Config:
        env_prefix = ''
        arbitrary_types_allowed = True
        protected_namespaces = ''

    def __init__(self, **data):
        super().__init__(**data)
        self.populate_from_env(self.provider_name)

    def populate_from_env(self, provider_name):
        env_prefix = provider_name.upper() + "_"
        for field_name, field_value in self.__fields__.items():
            if field_value.alias:
                env_var_name = env_prefix + field_value.alias.upper()
            else:
                env_var_name = env_prefix + field_name.upper()
            if env_var_name in os.environ:
                setattr(self, field_name, os.environ[env_var_name])
