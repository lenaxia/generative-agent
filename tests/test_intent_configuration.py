"""
Tests for Intent Processing Configuration

Tests the configuration updates for intent processing to ensure
the new architecture settings are properly loaded and validated.

Following TDD principles - tests written first.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml


class TestIntentProcessingConfiguration:
    """Test intent processing configuration loading and validation."""

    def test_load_intent_processing_config(self):
        """Test loading intent processing configuration from config.yaml."""
        # Load the actual config file
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify intent processing configuration exists
        assert "message_bus" in config
        assert "intent_processing" in config["message_bus"]

        intent_config = config["message_bus"]["intent_processing"]
        assert intent_config["enabled"] is True
        assert intent_config["validate_intents"] is True
        assert intent_config["timeout_seconds"] == 30
        assert intent_config["max_concurrent_intents"] == 50

    def test_load_single_file_roles_config(self):
        """Test loading single-file roles configuration."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify single-file roles configuration exists
        assert "role_system" in config
        assert "single_file_roles" in config["role_system"]

        single_file_config = config["role_system"]["single_file_roles"]
        assert single_file_config["enabled"] is True
        assert single_file_config["role_pattern"] == "*.py"
        assert single_file_config["auto_discovery"] is True
        assert single_file_config["validate_structure"] is True
        assert single_file_config["enforce_patterns"] is True

    def test_load_feature_flags_config(self):
        """Test loading updated feature flags configuration."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify feature flags include new intent processing flags
        assert "feature_flags" in config
        feature_flags = config["feature_flags"]

        assert feature_flags["enable_intent_processing"] is True
        assert feature_flags["enable_single_event_loop"] is True

        # Verify existing flags are preserved
        assert feature_flags["enable_universal_agent"] is True
        assert feature_flags["enable_task_scheduling"] is True
        assert feature_flags["enable_pause_resume"] is True
        assert feature_flags["enable_heartbeat"] is True

    def test_config_yaml_is_valid_yaml(self):
        """Test that config.yaml is valid YAML syntax."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Should be a dictionary
            assert isinstance(config, dict)
            assert len(config) > 0

        except yaml.YAMLError as e:
            pytest.fail(f"config.yaml contains invalid YAML syntax: {e}")

    def test_config_has_required_sections(self):
        """Test that config.yaml has all required sections."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Required top-level sections
        required_sections = [
            "framework",
            "llm_providers",
            "universal_agent",
            "role_system",
            "task_graph",
            "message_bus",
            "feature_flags",
        ]

        for section in required_sections:
            assert section in config, f"Missing required config section: {section}"

    def test_intent_processing_defaults(self):
        """Test intent processing configuration has sensible defaults."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        intent_config = config["message_bus"]["intent_processing"]

        # Verify sensible defaults
        assert intent_config["timeout_seconds"] > 0
        assert intent_config["timeout_seconds"] <= 60  # Not too long
        assert intent_config["max_concurrent_intents"] > 0
        assert intent_config["max_concurrent_intents"] <= 100  # Reasonable limit


class TestConfigurationIntegration:
    """Test configuration integration with components."""

    def test_create_temp_config_with_intent_processing(self):
        """Test creating temporary config with intent processing enabled."""
        temp_config = {
            "framework": {"type": "strands"},
            "message_bus": {
                "intent_processing": {
                    "enabled": True,
                    "validate_intents": True,
                    "timeout_seconds": 15,
                    "max_concurrent_intents": 25,
                }
            },
            "role_system": {
                "single_file_roles": {
                    "enabled": True,
                    "role_pattern": "*.py",
                    "auto_discovery": True,
                }
            },
            "feature_flags": {
                "enable_intent_processing": True,
                "enable_single_event_loop": True,
            },
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(temp_config, f)
            temp_config_path = f.name

        try:
            # Load and validate the temporary config
            with open(temp_config_path) as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config["message_bus"]["intent_processing"]["enabled"] is True
            assert loaded_config["role_system"]["single_file_roles"]["enabled"] is True
            assert loaded_config["feature_flags"]["enable_intent_processing"] is True

        finally:
            # Clean up temporary file
            os.unlink(temp_config_path)

    def test_config_backward_compatibility(self):
        """Test that configuration changes don't break existing functionality."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify existing configurations are preserved
        assert "llm_providers" in config
        assert "bedrock" in config["llm_providers"]
        assert "models" in config["llm_providers"]["bedrock"]

        # Verify role system still has original settings
        role_system = config["role_system"]
        assert role_system["roles_directory"] == "roles"
        assert role_system["registry"]["auto_refresh"] is True
        assert role_system["shared_tools"]["auto_discover"] is True

    def test_feature_flags_integration(self):
        """Test that new feature flags integrate properly with existing ones."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        feature_flags = config["feature_flags"]

        # New flags should be present
        assert "enable_intent_processing" in feature_flags
        assert "enable_single_event_loop" in feature_flags

        # Existing flags should be preserved
        assert "enable_universal_agent" in feature_flags
        assert "enable_task_scheduling" in feature_flags
        assert "enable_pause_resume" in feature_flags
        assert "enable_heartbeat" in feature_flags

        # All should be boolean values
        for flag_name, flag_value in feature_flags.items():
            assert isinstance(
                flag_value, bool
            ), f"Feature flag {flag_name} should be boolean, got {type(flag_value)}"


if __name__ == "__main__":
    pytest.main([__file__])
