from typing import Dict, Optional
from pydantic import BaseModel, validator, Field
from supervisor.metrics_manager import MetricsManager


class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_file_max_size: int = 1024 # 1 kB
    llm_providers: Dict[str, Dict] = {}
    disable_console_logging: Optional[bool] = None

from typing import Dict, Optional
from pydantic import BaseModel, validator

class LLMProviderConfig(BaseModel):
    name: str
    type: str
    model_name: Optional[str] = None
    registry_type: Optional[str] = None
    base_url: Optional[str] = None

class SupervisorConfig(BaseModel):
    logging: LoggingConfig = Field(..., description="Logging configuration")
    llm_providers: Dict[str, LLMProviderConfig] = {}

    max_retries: int = 3
    retry_delay: float = 0.1

    metrics_manager: Optional[MetricsManager] = Field(None, exclude=True)

    @validator('llm_providers', pre=True)
    def validate_llm_providers(cls, value):
        if not isinstance(value, dict):
            raise ValueError("llm_providers must be a dictionary")
        return {k: LLMProviderConfig(**v) for k, v in value.items()}

    def __init__(self, **data):
        super().__init__(**data)
        self.metrics_manager = MetricsManager()