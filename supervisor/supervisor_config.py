"""Supervisor configuration classes and settings management.

This module provides configuration classes and data structures for
managing supervisor settings, workflow parameters, and system
configuration options.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from supervisor.metrics_manager import MetricsManager


class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_file_max_size: int = 1024  # 1 kB
    llm_providers: dict[str, dict] = {}
    disable_console_logging: Optional[bool] = None


from typing import Dict, Optional

from pydantic import BaseModel


class LLMProviderConfig(BaseModel):
    name: str
    type: str


class MCPConfig(BaseModel):
    enabled: bool = False
    config_file: str = "config/mcp_config.yaml"


class FeatureFlags(BaseModel):
    enable_universal_agent: bool = True
    enable_mcp_integration: bool = False
    enable_task_scheduling: bool = True
    enable_pause_resume: bool = True
    enable_heartbeat: bool = True


class SupervisorConfig(BaseModel):
    logging: LoggingConfig = Field(..., description="Logging configuration")
    llm_providers: dict[str, dict] = {}
    agents: dict[str, dict] = {}
    mcp: Optional[MCPConfig] = Field(default_factory=lambda: MCPConfig())
    feature_flags: Optional[FeatureFlags] = Field(
        default_factory=lambda: FeatureFlags()
    )

    max_retries: int = 3
    retry_delay: float = 0.1

    metrics_manager: Optional[MetricsManager] = Field(None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.metrics_manager = MetricsManager()
