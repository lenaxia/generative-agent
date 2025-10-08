"""
Simplified Configuration Manager for Universal Agent System.

This module provides centralized configuration management with validation,
environment variable substitution, and backward compatibility with existing configs.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, path: str = ""):
        self.message = message
        self.path = path
        super().__init__(f"Configuration validation error at '{path}': {message}")


class ConfigManager:
    """
    Simplified configuration manager for Universal Agent system.

    Focuses on practical configuration loading, validation, and environment
    variable substitution while maintaining backward compatibility.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.config_data: dict[str, Any] = {}

    def load_config(self, validate: bool = True) -> dict[str, Any]:
        """
        Load configuration from file.

        Args:
            validate: Whether to validate configuration after loading

        Returns:
            Configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path) as f:
                self.config_data = yaml.safe_load(f) or {}

            # Substitute environment variables
            self.config_data = self._substitute_env_vars(self.config_data)

            # Validate configuration if requested
            if validate:
                self._validate_config()

            logger.info(f"Configuration loaded from: {self.config_path}")
            return self.config_data

        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Invalid YAML in config file: {e}", self.config_path
            )
        except Exception as e:
            raise ConfigValidationError(f"Error loading config: {e}", self.config_path)

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        if isinstance(config, dict):
            return {
                key: self._substitute_env_vars(value) for key, value in config.items()
            }
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Substitute ${VAR_NAME} and ${VAR_NAME:default} patterns
            pattern = r"\$\{([^}]+)\}"

            def replace_env_var(match):
                var_spec = match.group(1)
                if ":" in var_spec:
                    var_name, default_value = var_spec.split(":", 1)
                else:
                    var_name, default_value = var_spec, ""

                return os.getenv(var_name, default_value)

            return re.sub(pattern, replace_env_var, config)
        else:
            return config

    def _validate_config(self):
        """Validate configuration structure and values."""
        errors = []

        # Validate framework configuration
        framework_type = self.get_config("framework.type", "strands")
        if framework_type != "strands":
            errors.append(
                "framework.type must be 'strands' (legacy 'langchain' no longer supported)"
            )

        # Validate LLM providers
        llm_providers = self.get_config("llm_providers", {})
        if not llm_providers:
            errors.append("At least one LLM provider must be configured")

        # Validate Universal Agent role mapping
        role_mapping = self.get_config("universal_agent.role_llm_mapping", {})
        valid_llm_types = ["WEAK", "DEFAULT", "STRONG"]

        for role, llm_type in role_mapping.items():
            if llm_type not in valid_llm_types:
                errors.append(
                    f"Invalid LLM type '{llm_type}' for role '{role}'. Must be one of: {valid_llm_types}"
                )

        # Validate task management settings
        max_concurrent = self.get_config("task_management.max_concurrent_tasks", 5)
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
            errors.append(
                "task_management.max_concurrent_tasks must be a positive integer"
            )

        if errors:
            raise ConfigValidationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.

        Args:
            path: Dot-separated configuration path (e.g., "universal_agent.role_llm_mapping")
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        keys = path.split(".")
        current = self.config_data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set_config(self, path: str, value: Any):
        """
        Set configuration value by dot-separated path.

        Args:
            path: Dot-separated configuration path
            value: Value to set
        """
        keys = path.split(".")
        current = self.config_data

        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """
        Get feature flag value.

        Args:
            flag_name: Name of the feature flag
            default: Default value if flag not found

        Returns:
            Feature flag value
        """
        return self.get_config(f"feature_flags.{flag_name}", default)

    def get_framework_type(self) -> str:
        """Get configured framework type."""
        return self.get_config("framework.type", "strands")

    def is_universal_agent_enabled(self) -> bool:
        """Check if Universal Agent is enabled."""
        return self.get_feature_flag("enable_universal_agent", True)

    def is_mcp_enabled(self) -> bool:
        """Check if MCP integration is enabled."""
        return self.get_config("mcp.enabled", True)

    def get_mcp_config_file(self) -> str:
        """Get MCP configuration file path."""
        return self.get_config("mcp.config_file", "config/mcp_config.yaml")

    def get_llm_config(self, provider: str) -> dict[str, Any]:
        """
        Get LLM provider configuration.

        Args:
            provider: Provider name ("bedrock" or "openai")

        Returns:
            Provider configuration
        """
        return self.get_config(f"llm_providers.{provider}", {})

    def get_role_llm_mapping(self) -> dict[str, str]:
        """Get role to LLM type mapping."""
        return self.get_config(
            "universal_agent.role_llm_mapping",
            {
                "planning": "STRONG",
                "analysis": "STRONG",
                "coding": "STRONG",
                "search": "WEAK",
                "weather": "WEAK",
                "summarizer": "DEFAULT",
                "slack": "DEFAULT",
                "default": "DEFAULT",
            },
        )

    def get_log_level(self) -> str:
        """Get configured log level."""
        return self.get_config("logging.level", "INFO")

    def get_log_file(self) -> Optional[str]:
        """Get configured log file path."""
        return self.get_config("logging.log_file")

    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.get_config("development.debug_mode", False)

    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration file (defaults to original path)
        """
        save_path = output_path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(self.config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to: {save_path}")

    def reload_config(self):
        """Reload configuration from file."""
        self.load_config()

    def __str__(self) -> str:
        """String representation of configuration manager."""
        return f"ConfigManager(config_path={self.config_path}, sections={len(self.config_data)})"


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: str = "config.yaml") -> ConfigManager:
    """
    Get global configuration manager instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Global ConfigManager instance
    """
    global _config_manager
    if _config_manager is None or _config_manager.config_path != config_path:
        _config_manager = ConfigManager(config_path)
        _config_manager.load_config()
    return _config_manager


def reload_config():
    """Reload global configuration."""
    if _config_manager:
        _config_manager.reload_config()


def get_config(path: str, default: Any = None) -> Any:
    """
    Get configuration value using global config manager.

    Args:
        path: Dot-separated configuration path
        default: Default value if not found

    Returns:
        Configuration value
    """
    return get_config_manager().get_config(path, default)


def get_feature_flag(flag_name: str, default: bool = False) -> bool:
    """
    Get feature flag using global config manager.

    Args:
        flag_name: Feature flag name
        default: Default value

    Returns:
        Feature flag value
    """
    return get_config_manager().get_feature_flag(flag_name, default)


def get_framework_type() -> str:
    """Get framework type from global config."""
    return get_config_manager().get_framework_type()


def is_universal_agent_enabled() -> bool:
    """Check if Universal Agent is enabled."""
    return get_config_manager().is_universal_agent_enabled()


def get_role_llm_mapping() -> dict[str, str]:
    """Get role to LLM type mapping from global config."""
    return get_config_manager().get_role_llm_mapping()
