"""Supervisor configuration classes and settings management.

This module provides configuration classes and data structures for
managing supervisor settings, workflow parameters, and system
configuration options.
"""

from typing import Optional

from pydantic import BaseModel, Field

from supervisor.metrics_manager import MetricsManager


class LoggingConfig(BaseModel):
    """Configuration settings for supervisor logging system.

    Defines log levels, file output settings, and provider-specific
    logging configurations for the supervisor system.
    """

    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_file_max_size: int = 1024  # 1 kB
    llm_providers: dict[str, dict] = {}
    disable_console_logging: Optional[bool] = None
    loggers: Optional[dict[str, str]] = None


class LLMProviderConfig(BaseModel):
    """Configuration for individual LLM providers.

    Defines the name and type settings for specific LLM provider
    configurations within the supervisor system.
    """

    name: str
    type: str


class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) integration.

    Controls MCP feature enablement and configuration file location
    for external tool and resource integration.
    """

    enabled: bool = False
    config_file: str = "config/mcp_config.yaml"


class ArchitectureConfig(BaseModel):
    """Configuration for system architecture settings.

    Controls threading model, role system, and LLM development features
    from the threading architecture improvements (Documents 25, 26, 27).
    """

    threading_model: str = "single_event_loop"
    role_system: str = "single_file"
    llm_development: bool = True


class ProductionThreadingConfig(BaseModel):
    """Configuration for production threading settings."""

    heartbeat_interval: int = 30
    timer_check_interval: int = 5
    max_scheduled_tasks: int = 10
    task_timeout: int = 300


class ProductionMonitoringConfig(BaseModel):
    """Configuration for production monitoring settings."""

    track_intent_processing: bool = True
    log_handler_performance: bool = True
    validate_intent_schemas: bool = True


class ProductionConfig(BaseModel):
    """Configuration for production deployment settings."""

    threading: ProductionThreadingConfig = Field(
        default_factory=ProductionThreadingConfig
    )
    monitoring: ProductionMonitoringConfig = Field(
        default_factory=ProductionMonitoringConfig
    )


class IntentProcessingConfig(BaseModel):
    """Configuration for intent processing system."""

    enabled: bool = True
    validate_intents: bool = True
    timeout_seconds: int = 30


class FeatureFlags(BaseModel):
    """Feature flag configuration for supervisor capabilities.

    Controls the enablement of various supervisor features including
    universal agent support, MCP integration, and system capabilities.
    """

    enable_universal_agent: bool = True
    enable_mcp_integration: bool = False
    enable_task_scheduling: bool = True
    enable_pause_resume: bool = True
    enable_heartbeat: bool = True


class SupervisorConfig(BaseModel):
    """Main configuration class for the supervisor system.

    Aggregates all supervisor configuration settings including logging,
    LLM providers, agents, MCP integration, feature flags, and the new
    threading architecture settings from Documents 25, 26, 27.
    """

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

    # New threading architecture configuration (Phase 4)
    architecture: Optional[ArchitectureConfig] = Field(
        default_factory=lambda: ArchitectureConfig()
    )
    production: Optional[ProductionConfig] = Field(
        default_factory=lambda: ProductionConfig()
    )
    intent_processing: Optional[IntentProcessingConfig] = Field(
        default_factory=lambda: IntentProcessingConfig()
    )

    def __init__(self, **data):
        """Initialize SupervisorConfig with metrics manager.

        Args:
            **data: Configuration data for supervisor initialization.
        """
        super().__init__(**data)
        self.metrics_manager = MetricsManager()
