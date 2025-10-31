"""Tests for the summarizer role.

This test suite validates the summarizer role's ability to synthesize
information from multiple sources and present it in structured formats.
"""

import pytest


class TestSummarizerRole:
    """Test suite for summarizer role functionality."""

    def test_summarizer_role_registration(self):
        """Test that summarizer role is properly registered."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")

        assert "summarizer" in registry.roles
        summarizer = registry.roles["summarizer"]

        config = summarizer.config.get("role", {})
        assert config["name"] == "summarizer"
        assert config["llm_type"] == "DEFAULT"
        assert config["fast_reply"] is True
        assert "synthesize" in config["when_to_use"].lower()

    def test_summarizer_role_parameters(self):
        """Test that summarizer role has correct parameters."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")
        summarizer = registry.roles["summarizer"]

        config = summarizer.config.get("role", {})
        parameters = config.get("parameters", {})

        assert "format" in parameters
        assert "focus" in parameters
        assert "length" in parameters

        format_param = parameters["format"]
        assert format_param["type"] == "string"
        assert format_param["required"] is False
        assert "summary" in format_param["enum"]
        assert "itinerary" in format_param["enum"]

    def test_load_predecessor_results_function(self):
        """Test the load_predecessor_results pre-processing function."""
        from unittest.mock import Mock

        from roles.core_summarizer import load_predecessor_results

        context = Mock()

        instruction = "Previous task results available for context:\n- Task 1: Result 1\n- Task 2: Result 2\n\nCurrent task: Summarize the results"
        parameters = {"format": "summary", "length": "brief"}

        result = load_predecessor_results(instruction, context, parameters)

        assert "predecessor_results" in result
        assert "format" in result
        assert result["format"] == "summary"
        assert result["has_predecessors"] is True
        assert "included in your instruction" in result["predecessor_results"]

    def test_load_predecessor_results_no_results(self):
        """Test pre-processing when no predecessor results exist."""
        from unittest.mock import Mock

        from roles.core_summarizer import load_predecessor_results

        context = Mock()

        instruction = "Summarize the results"
        parameters = {}

        result = load_predecessor_results(instruction, context, parameters)

        assert "predecessor_results" in result
        assert "No predecessor results available" in result["predecessor_results"]
        assert result["has_predecessors"] is False

    def test_synthesis_intent_validation(self):
        """Test SynthesisIntent validation."""
        from roles.core_summarizer import SynthesisIntent

        valid_intent = SynthesisIntent(
            sources=[{"content": "test"}], format_type="summary"
        )
        assert valid_intent.validate() is True

        invalid_intent = SynthesisIntent(sources=[], format_type="invalid_format")
        assert invalid_intent.validate() is False

    def test_format_predecessor_results(self):
        """Test internal helper for formatting predecessor results."""
        from roles.core_summarizer import _format_predecessor_results

        results = [
            {"task_name": "Search Task", "result": "Found 10 items"},
            {"task_name": "Analysis Task", "result": "Analyzed data successfully"},
        ]

        formatted = _format_predecessor_results(results)

        assert "Search Task" in formatted
        assert "Analysis Task" in formatted
        assert "Found 10 items" in formatted
        assert "Analyzed data successfully" in formatted

    def test_summarizer_lifecycle_configuration(self):
        """Test that summarizer has correct lifecycle configuration."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")
        summarizer = registry.roles["summarizer"]

        config = summarizer.config.get("role", {})
        lifecycle = config.get("lifecycle", {})

        assert "pre_processing" in lifecycle
        assert lifecycle["pre_processing"]["enabled"] is True
        assert "load_predecessor_results" in lifecycle["pre_processing"]["functions"]

    def test_summarizer_system_prompt(self):
        """Test that summarizer has appropriate system prompt."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")
        summarizer = registry.roles["summarizer"]

        config = summarizer.config.get("role", {})
        prompts = config.get("prompts", {})
        system_prompt = prompts.get("system", "")

        assert "synthesis" in system_prompt.lower()
        assert "predecessor" in system_prompt.lower()
        assert "do not engage in casual conversation" in system_prompt.lower()
        assert "{{predecessor_results}}" in system_prompt
        assert "{{format}}" in system_prompt
