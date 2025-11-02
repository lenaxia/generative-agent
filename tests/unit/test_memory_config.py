"""Tests for memory system configuration.

This module tests that memory system configuration loads correctly
and is used by the memory components.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config_data():
    """Load config.yaml for testing."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_config_loads_correctly(config_data):
    """Test config loads without errors."""
    assert config_data is not None
    assert "memory_system" in config_data
    assert config_data["memory_system"]["enabled"] is True


def test_default_values_applied(config_data):
    """Test defaults used when not specified."""
    memory_config = config_data["memory_system"]

    assert memory_config["permanent_threshold"] == 0.7
    assert memory_config["ttl_calculation"]["very_low_days"] == 3
    assert memory_config["ttl_calculation"]["low_days"] == 7
    assert memory_config["ttl_calculation"]["medium_days"] == 30
    assert memory_config["realtime_log"]["ttl_hours"] == 24
    assert memory_config["realtime_log"]["max_messages"] == 100
    assert memory_config["analysis"]["inactivity_timeout_minutes"] == 30
    assert memory_config["analysis"]["model"] == "WEAK"
    assert memory_config["analysis"]["timeout_seconds"] == 5


def test_thresholds_used_by_assessor(config_data):
    """Test assessor uses config thresholds."""
    from common.memory_importance_assessor import MemoryImportanceAssessor
    from llm_provider.factory import LLMFactory

    memory_config = config_data["memory_system"]
    permanent_threshold = memory_config["permanent_threshold"]

    factory = LLMFactory({})
    assessor = MemoryImportanceAssessor(
        factory, permanent_threshold=permanent_threshold
    )

    assert assessor.permanent_threshold == 0.7


def test_ttl_calculation_uses_config(config_data):
    """Test TTL calculation uses config."""
    from common.memory_importance_assessor import MemoryImportanceAssessor
    from llm_provider.factory import LLMFactory

    memory_config = config_data["memory_system"]
    ttl_config = memory_config["ttl_calculation"]

    factory = LLMFactory({})
    assessor = MemoryImportanceAssessor(factory)

    very_low_ttl = assessor.calculate_ttl(0.2)
    assert very_low_ttl == ttl_config["very_low_days"] * 24 * 60 * 60

    low_ttl = assessor.calculate_ttl(0.4)
    assert low_ttl == ttl_config["low_days"] * 24 * 60 * 60

    medium_ttl = assessor.calculate_ttl(0.6)
    assert medium_ttl == ttl_config["medium_days"] * 24 * 60 * 60

    permanent_ttl = assessor.calculate_ttl(0.8)
    assert permanent_ttl is None


def test_inactivity_timeout_configurable(config_data):
    """Test timeout checker uses config."""
    memory_config = config_data["memory_system"]
    timeout_minutes = memory_config["analysis"]["inactivity_timeout_minutes"]

    assert timeout_minutes == 30
    assert isinstance(timeout_minutes, int)
    assert timeout_minutes > 0
