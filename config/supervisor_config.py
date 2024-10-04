from typing import Dict, Optional
from pydantic import BaseModel, validator
from llm_provider import LLMRegistry

class SupervisorConfig(BaseModel):
    log_level: str = "INFO"
    log_file: Optional[str] = None
    llm_providers: Dict[str, Dict] = {}

    @validator('llm_providers', pre=True)
    def validate_llm_providers(cls, value):
        if isinstance(value, dict):
            return value
        raise ValueError("llm_providers must be a dictionary")

    llm_registry: Optional[LLMRegistry] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.llm_registry = LLMRegistry()
