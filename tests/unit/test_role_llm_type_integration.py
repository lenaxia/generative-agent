"""Test integration of role LLM type mappings from YAML definitions."""

from unittest.mock import Mock, patch

from llm_provider.factory import LLMType
from llm_provider.role_registry import RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestRoleLLMTypeIntegration:
    """Test that role LLM types are correctly read from YAML definitions."""

    def test_role_registry_reads_llm_types_from_yaml(self):
        """Test that role registry can read LLM types from role definitions."""
        # Create a mock role registry with test data
        role_registry = RoleRegistry("roles")

        # Test that we can get LLM types for roles that should have them defined
        weather_llm_type = role_registry.get_role_llm_type("weather")
        assert (
            weather_llm_type == "WEAK"
        ), f"Expected WEAK for weather role, got {weather_llm_type}"

        planning_llm_type = role_registry.get_role_llm_type("planning")
        assert (
            planning_llm_type == "STRONG"
        ), f"Expected STRONG for planning role, got {planning_llm_type}"

        default_llm_type = role_registry.get_role_llm_type("default")
        assert (
            default_llm_type == "DEFAULT"
        ), f"Expected DEFAULT for default role, got {default_llm_type}"

    def test_role_registry_fallback_for_missing_role(self):
        """Test that role registry returns DEFAULT for non-existent roles."""
        role_registry = RoleRegistry("roles")

        nonexistent_llm_type = role_registry.get_role_llm_type("nonexistent_role")
        assert (
            nonexistent_llm_type == "DEFAULT"
        ), f"Expected DEFAULT for nonexistent role, got {nonexistent_llm_type}"

    def test_universal_agent_uses_role_registry(self):
        """Test that UniversalAgent uses role registry for LLM type determination."""
        # Create mock LLM factory
        mock_llm_factory = Mock()

        # Create role registry
        role_registry = RoleRegistry("roles")

        # Create universal agent
        universal_agent = UniversalAgent(
            llm_factory=mock_llm_factory, role_registry=role_registry
        )

        # Test that it correctly determines LLM types from role registry
        weather_llm_type = universal_agent._determine_llm_type_for_role("weather")
        assert (
            weather_llm_type == LLMType.WEAK
        ), f"Expected LLMType.WEAK for weather, got {weather_llm_type}"

        planning_llm_type = universal_agent._determine_llm_type_for_role("planning")
        assert (
            planning_llm_type == LLMType.STRONG
        ), f"Expected LLMType.STRONG for planning, got {planning_llm_type}"

        default_llm_type = universal_agent._determine_llm_type_for_role("default")
        assert (
            default_llm_type == LLMType.DEFAULT
        ), f"Expected LLMType.DEFAULT for default, got {default_llm_type}"

    def test_universal_agent_handles_invalid_llm_type(self):
        """Test that UniversalAgent handles invalid LLM types gracefully."""
        # Create mock role registry that returns invalid LLM type
        mock_role_registry = Mock()
        mock_role_registry.get_role_llm_type.return_value = "INVALID_TYPE"

        # Create mock LLM factory
        mock_llm_factory = Mock()

        # Create universal agent
        universal_agent = UniversalAgent(
            llm_factory=mock_llm_factory, role_registry=mock_role_registry
        )

        # Test that it falls back to DEFAULT for invalid types
        llm_type = universal_agent._determine_llm_type_for_role("test_role")
        assert (
            llm_type == LLMType.DEFAULT
        ), f"Expected LLMType.DEFAULT for invalid type, got {llm_type}"

    def test_get_all_role_llm_mappings(self):
        """Test that we can get all role LLM mappings at once."""
        role_registry = RoleRegistry("roles")

        all_mappings = role_registry.get_all_role_llm_mappings()

        # Verify it's a dictionary
        assert isinstance(
            all_mappings, dict
        ), f"Expected dict, got {type(all_mappings)}"

        # Verify some expected roles are present
        expected_roles = ["weather", "planning", "default", "search"]
        for role in expected_roles:
            if role in all_mappings:  # Only check if role exists in the system
                llm_type = all_mappings[role]
                assert llm_type in [
                    "WEAK",
                    "DEFAULT",
                    "STRONG",
                ], f"Invalid LLM type {llm_type} for role {role}"

    def test_role_registry_validates_llm_types(self):
        """Test that role registry validates LLM types."""
        # Create a mock role definition with invalid LLM type
        mock_role_def = Mock()
        mock_role_def.config = {"role": {"llm_type": "INVALID_TYPE"}}

        role_registry = RoleRegistry("roles")
        role_registry.roles = {"test_role": mock_role_def}

        # Should return DEFAULT for invalid type
        llm_type = role_registry.get_role_llm_type("test_role")
        assert (
            llm_type == "DEFAULT"
        ), f"Expected DEFAULT for invalid type, got {llm_type}"

    @patch("config.config_manager.logger")
    def test_config_manager_uses_role_registry(self, mock_logger):
        """Test that config manager can use role registry for mappings."""
        from config.config_manager import ConfigManager

        # Create config manager
        config_manager = ConfigManager("config.yaml")

        # Get role LLM mappings (should use role registry)
        mappings = config_manager.get_role_llm_mapping()

        # Verify it's a dictionary with expected structure
        assert isinstance(mappings, dict), f"Expected dict, got {type(mappings)}"

        # Verify all values are valid LLM types
        valid_types = {"WEAK", "DEFAULT", "STRONG"}
        for role, llm_type in mappings.items():
            assert (
                llm_type in valid_types
            ), f"Invalid LLM type {llm_type} for role {role}"
