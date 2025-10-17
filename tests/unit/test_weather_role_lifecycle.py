"""Tests for weather role lifecycle configuration and functions.

This test suite verifies that the weather role properly implements
the lifecycle hook system without hacks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roles.core_weather import (
    ROLE_CONFIG,
    fetch_weather_data,
    format_for_tts,
    pii_scrubber,
)


class TestWeatherRoleLifecycle:
    """Test weather role lifecycle configuration and functions."""

    def test_weather_role_config_has_lifecycle_section(self):
        """Test that weather role config includes lifecycle configuration."""
        assert (
            "lifecycle" in ROLE_CONFIG
        ), "Weather role must have lifecycle configuration"

        lifecycle_config = ROLE_CONFIG["lifecycle"]
        assert isinstance(
            lifecycle_config, dict
        ), "Lifecycle config must be a dictionary"

    def test_weather_role_has_pre_processing_config(self):
        """Test that weather role has proper pre-processing configuration."""
        lifecycle_config = ROLE_CONFIG.get("lifecycle", {})

        assert (
            "pre_processing" in lifecycle_config
        ), "Weather role must have pre-processing config"

        pre_config = lifecycle_config["pre_processing"]
        assert pre_config.get("enabled") is True, "Pre-processing must be enabled"
        assert "functions" in pre_config, "Pre-processing must specify functions"

        functions = pre_config["functions"]
        assert isinstance(functions, list), "Pre-processing functions must be a list"
        assert (
            "fetch_weather_data" in functions
        ), "Must include fetch_weather_data function"

    def test_weather_role_has_post_processing_config(self):
        """Test that weather role has proper post-processing configuration."""
        lifecycle_config = ROLE_CONFIG.get("lifecycle", {})

        assert (
            "post_processing" in lifecycle_config
        ), "Weather role must have post-processing config"

        post_config = lifecycle_config["post_processing"]
        assert post_config.get("enabled") is True, "Post-processing must be enabled"
        assert "functions" in post_config, "Post-processing must specify functions"

        functions = post_config["functions"]
        assert isinstance(functions, list), "Post-processing functions must be a list"
        assert "format_for_tts" in functions, "Must include format_for_tts function"
        assert "pii_scrubber" in functions, "Must include pii_scrubber function"

    @pytest.mark.asyncio
    async def test_fetch_weather_data_function_signature(self):
        """Test that fetch_weather_data has correct async signature."""
        import inspect

        # Function should be async
        assert inspect.iscoroutinefunction(
            fetch_weather_data
        ), "fetch_weather_data must be async function"

        # Check signature matches lifecycle pattern
        sig = inspect.signature(fetch_weather_data)
        params = list(sig.parameters.keys())

        expected_params = ["instruction", "context", "parameters"]
        assert (
            params == expected_params
        ), f"fetch_weather_data signature must be {expected_params}, got {params}"

    @pytest.mark.asyncio
    async def test_format_for_tts_function_signature(self):
        """Test that format_for_tts has correct async signature."""
        import inspect

        # Function should be async
        assert inspect.iscoroutinefunction(
            format_for_tts
        ), "format_for_tts must be async function"

        # Check signature matches lifecycle pattern
        sig = inspect.signature(format_for_tts)
        params = list(sig.parameters.keys())

        expected_params = ["llm_result", "context", "pre_data"]
        assert (
            params == expected_params
        ), f"format_for_tts signature must be {expected_params}, got {params}"

    @pytest.mark.asyncio
    async def test_pii_scrubber_function_signature(self):
        """Test that pii_scrubber has correct async signature."""
        import inspect

        # Function should be async
        assert inspect.iscoroutinefunction(
            pii_scrubber
        ), "pii_scrubber must be async function"

        # Check signature matches lifecycle pattern
        sig = inspect.signature(pii_scrubber)
        params = list(sig.parameters.keys())

        expected_params = ["llm_result", "context", "pre_data"]
        assert (
            params == expected_params
        ), f"pii_scrubber signature must be {expected_params}, got {params}"

    @pytest.mark.asyncio
    async def test_fetch_weather_data_execution(self):
        """Test that fetch_weather_data executes without errors."""
        # Mock parameters
        instruction = "What's the weather in Seattle?"
        context = MagicMock()
        parameters = {"location": "Seattle", "timeframe": "current"}

        # Mock the weather API calls
        with patch("roles.core_weather._city_to_coordinates") as mock_coords, patch(
            "roles.core_weather._check_weather"
        ) as mock_weather:
            mock_coords.return_value = {"lat": 47.6062, "lon": -122.3321}
            mock_weather.return_value = {
                "temperature": 65,
                "temperature_unit": "F",
                "short_forecast": "Partly Cloudy",
                "detailed_forecast": "Partly cloudy with light winds",
                "wind_speed": "5 mph",
                "wind_direction": "NW",
                "period_name": "This Afternoon",
            }

            # Execute function
            result = await fetch_weather_data(instruction, context, parameters)

            # Verify result structure
            assert isinstance(result, dict), "Result must be a dictionary"
            assert "weather_current" in result, "Result must contain weather_current"
            assert (
                "location_resolved" in result
            ), "Result must contain location_resolved"
            assert "coordinates" in result, "Result must contain coordinates"

    @pytest.mark.asyncio
    async def test_format_for_tts_execution(self):
        """Test that format_for_tts executes and formats text properly."""
        # Test input with markdown and technical terms
        llm_result = "The weather is **65°F** with *light winds* at 5 mph."
        context = MagicMock()
        pre_data = {"test": "data"}

        # Execute function
        result = await format_for_tts(llm_result, context, pre_data)

        # Verify formatting
        assert isinstance(result, str), "Result must be a string"
        assert "**" not in result, "Markdown formatting should be removed"
        assert "*" not in result, "Markdown formatting should be removed"
        assert "degrees Fahrenheit" in result, "°F should be expanded"
        assert "miles per hour" in result, "mph should be expanded"

    @pytest.mark.asyncio
    async def test_pii_scrubber_execution(self):
        """Test that pii_scrubber executes and removes sensitive data."""
        # Test input with coordinates and API keys
        llm_result = (
            "Weather at 47.606209,-122.332069 shows sunny skies. API_KEY=abc123def456"
        )
        context = MagicMock()
        pre_data = {"test": "data"}

        # Execute function
        result = await pii_scrubber(llm_result, context, pre_data)

        # Verify scrubbing
        assert isinstance(result, str), "Result must be a string"
        assert (
            "47.606209,-122.332069" not in result
        ), "Precise coordinates should be removed"
        assert "API_KEY=abc123def456" not in result, "API keys should be removed"
        assert (
            "[coordinates removed]" in result
        ), "Should indicate coordinates were removed"
        assert "[API key removed]" in result, "Should indicate API key was removed"

    def test_weather_hack_removal(self):
        """Test that the weather hack function is completely removed."""
        # Try to import the hack function - it should not exist
        try:
            from roles.core_weather import process_weather_request_with_data

            assert (
                False
            ), "process_weather_request_with_data hack function should be completely removed"
        except ImportError:
            # This is expected - the function should not exist
            pass

    def test_weather_role_no_execution_mode_complexity(self):
        """Test that weather role doesn't use ExecutionMode complexity."""
        import inspect

        # Check the role config
        role_config_source = inspect.getsource(lambda: ROLE_CONFIG)
        assert (
            "ExecutionMode" not in role_config_source
        ), "Weather role should not reference ExecutionMode"

        # Check lifecycle functions don't reference execution modes
        fetch_source = inspect.getsource(fetch_weather_data)
        assert (
            "ExecutionMode" not in fetch_source
        ), "fetch_weather_data should not reference ExecutionMode"

    def test_weather_role_uses_standard_lifecycle_pattern(self):
        """Test that weather role follows standard lifecycle patterns."""
        # Verify role config structure matches expected pattern
        assert ROLE_CONFIG["name"] == "weather", "Role name must be 'weather'"
        assert "tools" in ROLE_CONFIG, "Role must have tools configuration"
        assert "prompts" in ROLE_CONFIG, "Role must have prompts configuration"

        # Verify lifecycle configuration follows pattern
        lifecycle = ROLE_CONFIG["lifecycle"]

        # Pre-processing structure
        pre_config = lifecycle["pre_processing"]
        assert isinstance(pre_config["enabled"], bool), "enabled must be boolean"
        assert isinstance(pre_config["functions"], list), "functions must be list"

        # Post-processing structure
        post_config = lifecycle["post_processing"]
        assert isinstance(post_config["enabled"], bool), "enabled must be boolean"
        assert isinstance(post_config["functions"], list), "functions must be list"


class TestWeatherRoleIntegration:
    """Test weather role integration with lifecycle system."""

    def test_weather_role_discoverable_by_registry(self):
        """Test that weather role can be discovered by RoleRegistry."""
        from llm_provider.role_registry import RoleRegistry

        # Create registry and load roles
        registry = RoleRegistry("roles")

        # Verify weather role is loaded
        weather_role = registry.get_role("weather")
        assert weather_role is not None, "Weather role must be discoverable by registry"

        # Verify lifecycle functions are discovered
        lifecycle_functions = registry.get_lifecycle_functions("weather")
        assert isinstance(
            lifecycle_functions, dict
        ), "Lifecycle functions must be a dict"

        # Verify expected functions are present
        expected_functions = ["fetch_weather_data", "format_for_tts", "pii_scrubber"]
        for func_name in expected_functions:
            assert (
                func_name in lifecycle_functions
            ), f"Function {func_name} must be discoverable by registry"

    def test_weather_lifecycle_functions_are_callable(self):
        """Test that discovered lifecycle functions are callable."""
        from llm_provider.role_registry import RoleRegistry

        registry = RoleRegistry("roles")
        lifecycle_functions = registry.get_lifecycle_functions("weather")

        # Test each function is callable
        for func_name, func in lifecycle_functions.items():
            assert callable(func), f"Function {func_name} must be callable"

            # Test async functions
            if func_name in ["fetch_weather_data", "format_for_tts", "pii_scrubber"]:
                import inspect

                assert inspect.iscoroutinefunction(
                    func
                ), f"Function {func_name} must be async"
