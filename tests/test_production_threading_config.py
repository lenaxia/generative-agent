"""
Production Threading Configuration Tests

This module tests the production configuration for the threading architecture
improvements implemented in Documents 25, 26, and 27.

Created: 2025-10-13
Part of: Phase 4 - Production Deployment (Document 27)
"""

import logging
from unittest.mock import patch

import pytest
import yaml

from supervisor.config_manager import ConfigManager
from supervisor.supervisor import Supervisor
from supervisor.threading_monitor import (
    get_threading_monitor,
    validate_threading_architecture,
)

logger = logging.getLogger(__name__)


class TestProductionThreadingConfig:
    """Test production threading configuration."""

    def test_production_config_structure(self):
        """Test that production config has required threading sections."""
        # Load config
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Verify architecture section
        assert "architecture" in config, "Config should have architecture section"
        arch_config = config["architecture"]
        assert (
            arch_config["threading_model"] == "single_event_loop"
        ), "Should use single event loop"
        assert (
            arch_config["llm_development"] is True
        ), "Should enable LLM development mode"

        # Verify production section
        assert "production" in config, "Config should have production section"
        prod_config = config["production"]

        # Verify threading subsection
        assert "threading" in prod_config, "Production should have threading config"
        threading_config = prod_config["threading"]
        assert (
            "heartbeat_interval" in threading_config
        ), "Should have heartbeat interval"
        assert (
            "timer_check_interval" in threading_config
        ), "Should have timer check interval"
        assert (
            "max_scheduled_tasks" in threading_config
        ), "Should have max scheduled tasks"
        assert "task_timeout" in threading_config, "Should have task timeout"

        # Verify monitoring subsection
        assert "monitoring" in prod_config, "Production should have monitoring config"
        monitoring_config = prod_config["monitoring"]
        assert (
            monitoring_config["track_intent_processing"] is True
        ), "Should track intent processing"
        assert (
            monitoring_config["log_handler_performance"] is True
        ), "Should log handler performance"
        assert (
            monitoring_config["validate_intent_schemas"] is True
        ), "Should validate intent schemas"

    def test_intent_processing_config(self):
        """Test intent processing configuration."""
        # Load config
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Verify intent processing section
        assert (
            "intent_processing" in config
        ), "Config should have intent processing section"
        intent_config = config["intent_processing"]
        assert intent_config["enabled"] is True, "Intent processing should be enabled"
        assert (
            intent_config["validate_intents"] is True
        ), "Intent validation should be enabled"
        assert "timeout_seconds" in intent_config, "Should have timeout configuration"

    def test_role_system_config(self):
        """Test role system configuration for single-file roles."""
        # Load config
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Verify role system section
        assert "role_system" in config, "Config should have role system section"
        role_config = config["role_system"]

        # Verify single-file role configuration
        assert role_config["roles_directory"] == "roles", "Should use roles directory"
        assert role_config["role_pattern"] == "*.py", "Should look for Python files"
        assert role_config["auto_discovery"] is True, "Should enable auto discovery"
        assert (
            role_config["validate_role_structure"] is True
        ), "Should validate role structure"
        assert role_config["enforce_patterns"] is True, "Should enforce patterns"

    def test_config_manager_loads_threading_config(self):
        """Test that ConfigManager properly loads threading configuration."""
        config_manager = ConfigManager("config.yaml")
        config = config_manager.load_config()

        # Verify threading configuration is loaded
        assert hasattr(
            config, "architecture"
        ), "Config should have architecture attribute"
        assert hasattr(config, "production"), "Config should have production attribute"
        assert hasattr(
            config, "intent_processing"
        ), "Config should have intent processing attribute"

    def test_threading_architecture_validation(self):
        """Test threading architecture validation function."""
        # Test validation function
        is_valid = validate_threading_architecture()

        # Should be valid (single thread)
        assert is_valid is True, "Threading architecture should be valid"

    def test_threading_monitor_initialization(self):
        """Test threading monitor can be initialized."""
        # Get monitor instance
        monitor = get_threading_monitor()

        # Verify monitor is properly initialized
        assert monitor is not None, "Monitor should be initialized"

        # Test health reporting
        health = monitor.get_threading_health()
        assert isinstance(health, dict), "Health should be a dictionary"
        assert "thread_count" in health, "Should include thread count"
        assert "main_thread_only" in health, "Should include main thread validation"
        assert "health_status" in health, "Should include health status"

    def test_production_config_values_are_reasonable(self):
        """Test that production configuration values are reasonable."""
        # Load config
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        prod_config = config["production"]
        threading_config = prod_config["threading"]

        # Verify reasonable values
        assert (
            10 <= threading_config["heartbeat_interval"] <= 300
        ), "Heartbeat interval should be reasonable"
        assert (
            1 <= threading_config["timer_check_interval"] <= 60
        ), "Timer check interval should be reasonable"
        assert (
            1 <= threading_config["max_scheduled_tasks"] <= 100
        ), "Max scheduled tasks should be reasonable"
        assert (
            30 <= threading_config["task_timeout"] <= 3600
        ), "Task timeout should be reasonable"

    def test_supervisor_respects_production_config(self):
        """Test that Supervisor respects production configuration."""
        # Create supervisor with config
        supervisor = Supervisor("config.yaml")

        # Verify single event loop configuration is applied
        assert supervisor._use_single_event_loop is True, "Should use single event loop"
        assert isinstance(
            supervisor._scheduled_tasks, list
        ), "Should initialize scheduled tasks"

    def test_feature_flags_compatibility(self):
        """Test that feature flags are compatible with threading architecture."""
        # Load config
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        # Verify feature flags exist and are compatible
        if "feature_flags" in config:
            feature_flags = config["feature_flags"]

            # These features should be compatible with threading architecture
            compatible_features = [
                "enable_universal_agent",
                "enable_mcp_integration",
                "enable_task_scheduling",
                "enable_pause_resume",
                "enable_heartbeat",
            ]

            for feature in compatible_features:
                if feature in feature_flags:
                    # Just verify they can be loaded (compatibility test)
                    assert isinstance(
                        feature_flags[feature], bool
                    ), f"Feature {feature} should be boolean"

    def test_config_validation_with_threading_architecture(self):
        """Test configuration validation with threading architecture."""
        # This test ensures the config is valid and can be used by the system
        try:
            supervisor = Supervisor("config.yaml")

            # Basic validation - should not raise exceptions
            assert (
                supervisor.config_manager is not None
            ), "Config manager should be initialized"
            assert (
                supervisor._use_single_event_loop is True
            ), "Single event loop should be enabled"

            # Configuration should be accessible
            config = supervisor.config_manager.load_config()
            assert config is not None, "Config should be loaded"

        except Exception as e:
            pytest.fail(f"Configuration validation failed: {e}")
