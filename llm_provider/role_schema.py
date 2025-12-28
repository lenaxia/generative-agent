"""Pydantic schema for role definition validation.

This module provides schema validation for role definition YAML files to ensure
they have the correct structure and prevent runtime errors.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class ToolsConfig(BaseModel):
    """Schema for tools configuration."""

    automatic: bool = False
    shared: list[str] = Field(default_factory=list)
    role_specific: list[str] = Field(default_factory=list, alias="role_specific")
    fast_reply: dict[str, Any] | None = None
    mcp_keywords: list[str] | None = None

    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class RoleMetadata(BaseModel):
    """Schema for role metadata section."""

    name: str
    version: str = "1.0.0"
    description: str
    fast_reply: bool = False
    llm_type: str = "DEFAULT"
    author: str | None = None
    when_to_use: str | None = None

    class Config:
        extra = "allow"


class ParameterSchema(BaseModel):
    """Schema for individual parameter definitions."""

    type: str
    required: bool = False
    description: str | None = None
    examples: list[str] | None = None
    enum: list[str] | None = None

    class Config:
        extra = "allow"


class EventDataSchema(BaseModel):
    """Schema for event data structure."""

    class Config:
        extra = "allow"  # Events can have flexible schemas


class EventDefinition(BaseModel):
    """Schema for event definitions."""

    event_type: str
    description: str | None = None
    handler: str | None = None
    data_schema: dict[str, Any] | None = None
    condition: str | None = None
    example: dict[str, Any] | None = None

    class Config:
        extra = "allow"


class EventsConfig(BaseModel):
    """Schema for events configuration."""

    publishes: list[EventDefinition] = Field(default_factory=list)
    subscribes: list[EventDefinition] = Field(default_factory=list)

    class Config:
        extra = "allow"


class LifecycleFunction(BaseModel):
    """Schema for lifecycle function definitions."""

    name: str
    description: str | None = None
    uses_parameters: list[str] | None = None

    class Config:
        extra = "allow"


class LifecyclePhase(BaseModel):
    """Schema for lifecycle phase configuration."""

    enabled: bool = True
    functions: list[LifecycleFunction] = Field(default_factory=list)
    data_injection: bool = False

    class Config:
        extra = "allow"


class LifecycleConfig(BaseModel):
    """Schema for lifecycle configuration."""

    pre_processing: LifecyclePhase | None = None
    post_processing: LifecyclePhase | None = None

    class Config:
        extra = "allow"


class PromptsConfig(BaseModel):
    """Schema for prompts configuration."""

    system: str = "You are a helpful AI assistant."
    user: str | None = None

    class Config:
        extra = "allow"


class ModelConfig(BaseModel):
    """Schema for model configuration."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    max_context: int | None = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        extra = "allow"


class CapabilitiesConfig(BaseModel):
    """Schema for capabilities configuration."""

    max_iterations: int = Field(default=5, gt=0)
    timeout_seconds: int = Field(default=300, gt=0)

    class Config:
        extra = "allow"


class LoggingConfig(BaseModel):
    """Schema for logging configuration."""

    level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    include_tool_calls: bool = True

    class Config:
        extra = "allow"


class RoleDefinitionSchema(BaseModel):
    """Complete schema for role definition YAML files."""

    # Core fields that map to the current RoleDefinition dataclass
    name: str | None = None  # Can be inferred from role.name or filename
    description: str | None = None  # Can be inferred from role.description
    version: str | None = "1.0.0"
    llm_type: str | None = "DEFAULT"

    # Structured sections
    role: RoleMetadata | None = None
    parameters: dict[str, ParameterSchema] = Field(default_factory=dict)
    events: EventsConfig | None = None
    lifecycle: LifecycleConfig | None = None
    prompts: PromptsConfig | None = None
    model_configuration: ModelConfig | None = Field(default=None, alias="model_config")
    tools: ToolsConfig | None = None
    capabilities: CapabilitiesConfig | None = None
    logging: LoggingConfig | None = None

    # Legacy fields for backward compatibility
    lifecycle_functions: list[str] = Field(default_factory=list)
    config: dict[str, Any] | None = None

    @validator("tools")
    def validate_tools_is_dict(cls, v):
        """Ensure tools is always a dict, not a list."""
        if isinstance(v, list):
            raise ValueError(
                f"tools must be a dictionary, not a list. "
                f"Found: {v}. "
                f"Use 'tools: {{automatic: false, shared: []}}' instead of 'tools: []'"
            )
        return v

    @validator("name", pre=True, always=True)
    def set_name(cls, v, values):
        """Set name from role.name if not provided directly."""
        if v is None and "role" in values and values["role"]:
            return values["role"].name
        return v

    @validator("description", pre=True, always=True)
    def set_description(cls, v, values):
        """Set description from role.description if not provided directly."""
        if v is None and "role" in values and values["role"]:
            return values["role"].description
        return v

    class Config:
        extra = "allow"  # Allow additional fields for extensibility
        validate_assignment = True


def validate_role_definition(data: dict[str, Any]) -> RoleDefinitionSchema:
    """Validate a role definition dictionary against the schema.

    Args:
        data: Raw role definition data from YAML

    Returns:
        Validated RoleDefinitionSchema instance

    Raises:
        ValidationError: If the data doesn't match the schema
    """
    return RoleDefinitionSchema(**data)
