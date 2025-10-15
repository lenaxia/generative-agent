"""
Production Deployment Tests

Tests for production-ready deployment of the threading architecture improvements.
Validates production configuration, monitoring, and final integration.

Following TDD principles with production-focused testing.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from common.intent_processor import IntentProcessor
from common.message_bus import MessageBus
from supervisor.supervisor import Supervisor


class TestProductionConfiguration:
    """Test production configuration for threading architecture."""

    def test_production_config_validation(self):
        """Test that production config has all required sections."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Production-ready sections
        required_sections = [
            "framework",
            "llm_providers",
            "role_system",
            "message_bus",
            "feature_flags",
            "logging",
            "heartbeat",
        ]

        for section in required_sections:
            assert section in config, f"Missing required production section: {section}"

    def test_intent_processing_production_config(self):
        """Test intent processing production configuration."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Intent processing should be enabled
        intent_config = config["message_bus"]["intent_processing"]
        assert intent_config["enabled"] is True
        assert intent_config["validate_intents"] is True
        assert intent_config["timeout_seconds"] > 0
        assert intent_config["max_concurrent_intents"] > 0

    def test_single_file_roles_production_config(self):
        """Test single-file roles production configuration."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Single-file roles should be enabled
        role_config = config["role_system"]["single_file_roles"]
        assert role_config["enabled"] is True
        assert role_config["auto_discovery"] is True
        assert role_config["validate_structure"] is True

    def test_feature_flags_production_ready(self):
        """Test that feature flags are production-ready."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        feature_flags = config["feature_flags"]

        # Threading architecture flags should be enabled
        assert feature_flags["enable_intent_processing"] is True
        assert feature_flags["enable_single_event_loop"] is True

        # Core system flags should be enabled
        assert feature_flags["enable_universal_agent"] is True
        assert feature_flags["enable_heartbeat"] is True


class TestProductionMonitoring:
    """Test production monitoring capabilities."""

    def test_intent_processor_metrics(self):
        """Test intent processor provides production metrics."""
        processor = IntentProcessor()

        # Should have metrics methods
        assert hasattr(processor, "get_processed_count")
        assert hasattr(processor, "get_registered_handlers")

        # Metrics should be accessible
        count = processor.get_processed_count()
        handlers = processor.get_registered_handlers()

        assert isinstance(count, int)
        assert isinstance(handlers, dict)

    def test_message_bus_metrics(self):
        """Test MessageBus provides production metrics."""
        bus = MessageBus()
        bus.start()

        # Should have metrics methods
        assert hasattr(bus, "get_intent_processor_metrics")

        # Metrics should be accessible
        metrics = bus.get_intent_processor_metrics()

        assert isinstance(metrics, dict)
        assert "processed_count" in metrics
        assert "registered_handlers" in metrics

    def test_supervisor_status_monitoring(self, tmp_path):
        """Test Supervisor status monitoring for production."""
        config_content = """
framework:
  type: "strands"
llm_providers:
  bedrock:
    models:
      DEFAULT: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
logging:
  level: "INFO"
  log_file: "logs/production.log"
feature_flags:
  enable_single_event_loop: true
  enable_intent_processing: true
"""
        config_file = tmp_path / "prod_config.yaml"
        config_file.write_text(config_content)

        supervisor = Supervisor(str(config_file))

        # Should have status method
        assert hasattr(supervisor, "status")

        # Status should be callable
        status = supervisor.status()

        # Status should contain production-relevant information
        if status:
            assert isinstance(status, dict)


class TestProductionReliability:
    """Test production reliability features."""

    def test_error_handling_production_ready(self):
        """Test error handling is production-ready."""
        from common.enhanced_event_context import create_context_from_event_data
        from roles.core_timer import handle_timer_expiry

        # Test with various error conditions
        error_cases = [
            None,  # None data
            [],  # Empty list
            {},  # Empty dict
            "invalid",  # Invalid string
            {"malformed": None},  # Malformed dict
        ]

        for error_data in error_cases:
            context = create_context_from_event_data(
                error_data, source="production_test"
            )

            # Should not raise exceptions
            intents = handle_timer_expiry(error_data, context)

            # Should return valid intents
            assert isinstance(intents, list)
            assert all(intent.validate() for intent in intents)

    def test_system_recovery_production(self):
        """Test system recovery capabilities for production."""
        bus = MessageBus()
        bus.start()

        # Should be able to enable/disable intent processing
        assert hasattr(bus, "enable_intent_processing")

        # Should handle enable/disable gracefully
        bus.enable_intent_processing(False)
        assert bus._enable_intent_processing is False

        bus.enable_intent_processing(True)
        assert bus._enable_intent_processing is True

    def test_configuration_hot_reload_ready(self, tmp_path):
        """Test configuration is ready for hot reload."""
        config_content = """
framework:
  type: "strands"
llm_providers:
  bedrock:
    models:
      DEFAULT: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
logging:
  level: "INFO"
  log_file: "logs/production.log"
message_bus:
  intent_processing:
    enabled: true
    validate_intents: true
    timeout_seconds: 30
feature_flags:
  enable_intent_processing: true
  enable_single_event_loop: true
"""
        config_file = tmp_path / "hot_reload_config.yaml"
        config_file.write_text(config_content)

        # Should be able to create supervisor with config
        supervisor = Supervisor(str(config_file))

        # Configuration should be loaded
        assert supervisor.config is not None


class TestProductionDeployment:
    """Test production deployment readiness."""

    def test_all_components_production_ready(self):
        """Test that all components are production-ready."""
        # Intent system
        from common.intents import NotificationIntent

        intent = NotificationIntent(message="Production test", channel="prod")
        assert intent.validate()

        # Intent processor
        processor = IntentProcessor()
        assert hasattr(processor, "get_processed_count")

        # Enhanced context
        from common.enhanced_event_context import LLMSafeEventContext

        context = LLMSafeEventContext(source="production")
        assert context.is_valid()

        # Single-file timer role
        from roles.core_timer import register_role

        role_info = register_role()
        assert "config" in role_info

    def test_backward_compatibility_production(self):
        """Test backward compatibility for production deployment."""
        # MessageBus should support both old and new methods
        bus = MessageBus()

        # Legacy methods should exist
        assert hasattr(bus, "publish")
        assert hasattr(bus, "subscribe")
        assert hasattr(bus, "start")
        assert hasattr(bus, "stop")

        # New methods should exist
        assert hasattr(bus, "publish_async")
        assert hasattr(bus, "set_dependencies")

    def test_logging_production_ready(self, tmp_path):
        """Test logging is production-ready."""
        config_content = """
framework:
  type: "strands"
llm_providers:
  bedrock:
    models:
      DEFAULT: "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
logging:
  level: "INFO"
  log_file: "logs/production.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""
        config_file = tmp_path / "logging_config.yaml"
        config_file.write_text(config_content)

        # Should load logging configuration
        with open(config_file) as f:
            config = yaml.safe_load(f)

        logging_config = config["logging"]
        assert logging_config["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert "log_file" in logging_config
        assert "format" in logging_config

    def test_security_considerations_production(self):
        """Test security considerations for production."""
        # Intent validation should prevent malicious intents
        from common.intents import AuditIntent, NotificationIntent

        # Invalid intents should be rejected
        invalid_notification = NotificationIntent(message="", channel="test")
        assert invalid_notification.validate() is False

        invalid_audit = AuditIntent(action="", details={})
        assert invalid_audit.validate() is False

        # Valid intents should pass
        valid_notification = NotificationIntent(message="Valid", channel="test")
        assert valid_notification.validate() is True

    def test_performance_production_ready(self):
        """Test performance characteristics are production-ready."""
        from common.enhanced_event_context import create_context_from_event_data
        from roles.core_timer import handle_timer_expiry

        # Should handle reasonable load
        context = create_context_from_event_data(
            ["timer_123", "Production test"], source="timer"
        )

        # Process multiple events quickly
        import time

        start_time = time.time()

        for i in range(50):
            intents = handle_timer_expiry(["timer_123", "Production test"], context)
            assert len(intents) > 0

        processing_time = time.time() - start_time

        # Should be fast enough for production
        assert (
            processing_time < 1.0
        ), f"Production processing too slow: {processing_time:.4f}s"


if __name__ == "__main__":
    pytest.main([__file__])
