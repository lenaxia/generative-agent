from typing import Dict, Optional
from pydantic import BaseModel, validator, Field
from supervisor.metrics_manager import MetricsManager

class SupervisorConfig(BaseModel):
    log_level: str = "INFO"
    log_file: Optional[str] = None
    llm_providers: Dict[str, Dict] = {}

    max_retries: int = 3
    retry_delay: float = 0.1

    metrics_manager: Optional[MetricsManager] = Field(None, exclude=True)

    @validator('llm_providers', pre=True)
    def validate_llm_providers(cls, value):
        if isinstance(value, dict):
            return value
        raise ValueError("llm_providers must be a dictionary")


    def __init__(self, **data):
        super().__init__(**data)

        self.metrics_manager = MetricsManager()

