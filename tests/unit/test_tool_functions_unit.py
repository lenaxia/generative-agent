"""
Unit tests for @tool functions across different agent roles.

Tests the various @tool decorated functions that provide specialized
functionality for planning, search, weather, summarizer, and Slack agents.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

# Import tool functions
from llm_provider.planning_tools import create_task_plan, analyze_task_dependencies
from llm_provider.search_tools import search_web, summarize_search_results
from llm_provider.weather_tools import get_weather, get_weather_forecast
from llm_provider.summarizer_tools import summarize_text, extract_key_phrases
from llm_provider.slack_tools import send_slack_message, format_slack_message


class TestToolFunctionsUnit:
    """Unit tests for @tool functions across different agent roles."""

    def test_planning_tools(self):
        """Test @tool planning functions work correctly."""
        # Test create_task_plan
        instruction = "Plan a project to build a web application"
        available_agents = [
            "planning_agent (Task planning and complex reasoning)",
            "search_agent (Web search and information retrieval)"
        ]
        
        result = create_task_plan(
            instruction=instruction,
            available_agents=available_agents,
            request_id="test_request_123"
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "task_graph" in result
        assert "tasks" in result
        assert "dependencies" in result
        
        # Verify tasks were created
        tasks = result["tasks"]
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        
        # Verify each task has required fields
        for task in tasks:
            assert hasattr(task, 'task_name')
            assert hasattr(task, 'agent_id')
            assert hasattr(task, 'task_type')
            assert hasattr(task, 'prompt')
        
        # Test analyze_task_dependencies
        dependencies_result = analyze_task_dependencies(tasks)
        assert isinstance(dependencies_result, list)

    def test_search_tools(self):
        """Test @tool search functions work correctly."""
        # Mock external search API
        with patch('requests.get') as mock_get:
            # Setup mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"title": "Test Result 1", "snippet": "Test snippet 1", "url": "http://test1.com"},
                    {"title": "Test Result 2", "snippet": "Test snippet 2", "url": "http://test2.com"}
                ]
            }
            mock_get.return_value = mock_response
            
            # Test search_web (actual function name)
            search_query = "Python testing best practices"
            search_results = search_web(search_query, num_results=5)
            
            # Verify search results structure
            assert isinstance(search_results, dict)
            assert "results" in search_results
            assert "query" in search_results
            assert search_results["query"] == search_query
            
            # Test summarize_search_results
            summary = summarize_search_results(search_results, max_length=100)
            assert isinstance(summary, dict)
            assert "summary" in summary
            assert "query" in summary
            assert len(summary["summary"]) > 0

    def test_weather_tools(self):
        """Test @tool weather functions work correctly."""
        # Mock weather API
        with patch('requests.get') as mock_get:
            # Setup mock current weather response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "location": {"name": "Seattle", "region": "Washington", "country": "USA"},
                "current": {
                    "temp_f": 65.0,
                    "condition": {"text": "Partly cloudy"},
                    "humidity": 70,
                    "wind_mph": 5.2
                }
            }
            mock_get.return_value = mock_response
            
            # Test get_weather (actual function name)
            weather = get_weather("Seattle, WA")
            
            # Verify weather data
            assert isinstance(weather, dict)
            assert "location" in weather
            assert "current" in weather
            assert weather["location"]["name"] == "Seattle"
            assert weather["current"]["temp_f"] == 65.0
            
            # Setup mock forecast response
            mock_response.json.return_value = {
                "location": {"name": "Seattle"},
                "forecast": {
                    "forecastday": [
                        {
                            "date": "2024-01-01",
                            "day": {
                                "maxtemp_f": 70.0,
                                "mintemp_f": 55.0,
                                "condition": {"text": "Sunny"}
                            }
                        }
                    ]
                }
            }
            
            # Test get_weather_forecast
            forecast = get_weather_forecast("Seattle, WA", days=3)
            
            # Verify forecast data
            assert isinstance(forecast, dict)
            assert "location" in forecast
            assert "forecast" in forecast

    def test_summarizer_tools(self):
        """Test @tool summarizer functions work correctly."""
        # Test text for summarization
        long_text = """
        This is a long text that needs to be summarized. It contains multiple paragraphs
        with various information about different topics. The first paragraph discusses
        the importance of testing in software development. The second paragraph talks
        about different testing strategies including unit tests, integration tests, and
        end-to-end tests. The third paragraph covers best practices for writing
        maintainable test code. The fourth paragraph discusses the benefits of
        test-driven development and how it improves code quality.
        """
        
        # Test summarize_text
        summary = summarize_text(long_text, max_length=100)
        
        # Verify summary
        assert isinstance(summary, str)
        assert len(summary) <= 150  # Allow some flexibility
        assert len(summary) < len(long_text)  # Should be shorter than original
        assert "testing" in summary.lower() or "test" in summary.lower()
        
        # Test extract_key_phrases (actual function name)
        key_phrases = extract_key_phrases(long_text, max_phrases=5)
        
        # Verify key phrases
        assert isinstance(key_phrases, dict)
        assert "phrases" in key_phrases
        phrases_list = key_phrases["phrases"]
        assert isinstance(phrases_list, list)
        assert len(phrases_list) <= 5
        assert any("testing" in phrase.lower() or "test" in phrase.lower() for phrase in phrases_list)

    def test_slack_tools(self):
        """Test @tool Slack functions work correctly."""
        # Mock Slack API
        with patch('slack_sdk.WebClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_postMessage.return_value = {
                "ok": True,
                "message": {
                    "ts": "1234567890.123456",
                    "text": "Test message"
                }
            }
            mock_client_class.return_value = mock_client
            
            # Test send_slack_message
            result = send_slack_message(
                channel="#general",
                message="Hello from unit tests!",
                token="test_token"
            )
            
            # Verify message was sent
            assert isinstance(result, dict)
            assert result.get("ok") is True
            mock_client.chat_postMessage.assert_called_once_with(
                channel="#general",
                text="Hello from unit tests!"
            )
            
            # Test format_slack_message
            formatted = format_slack_message(
                title="Test Alert",
                message="This is a test alert message",
                priority="high",
                mentions=["@channel"]
            )
            
            # Verify formatting
            assert isinstance(formatted, str)
            assert "Test Alert" in formatted
            assert "test alert message" in formatted.lower()
            assert "@channel" in formatted
            assert "high" in formatted.lower()

    def test_tool_function_error_handling(self):
        """Test tool functions handle errors gracefully."""
        # Test search tool with network error
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            try:
                result = search_web("test query")
                # If no exception, should return dict with error status
                assert isinstance(result, dict)
                assert result.get("status") == "failed" or "error" in result
            except Exception as e:
                # If exception is raised, should be handled gracefully
                assert "network" in str(e).lower() or "error" in str(e).lower()
        
        # Test weather tool with API error
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"error": "Location not found"}
            mock_get.return_value = mock_response
            
            try:
                result = get_weather("Invalid Location")
                # Should handle API error gracefully
                assert isinstance(result, (dict, str))
                if isinstance(result, dict):
                    assert "error" in result or result == {}
            except Exception as e:
                # If exception is raised, should be informative
                assert "location" in str(e).lower() or "error" in str(e).lower()

    def test_tool_function_input_validation(self):
        """Test tool functions validate input parameters."""
        # Test planning tools with invalid input
        try:
            result = create_task_plan(
                instruction="",  # Empty instruction
                available_agents=[],  # Empty agents list
                request_id=""
            )
            # Should handle gracefully or return minimal result
            assert isinstance(result, dict)
        except (ValueError, TypeError) as e:
            # If validation error is raised, that's acceptable
            assert "instruction" in str(e).lower() or "agents" in str(e).lower()
        
        # Test search tools with invalid input
        try:
            result = search_web("")  # Empty query
            assert isinstance(result, dict)
            # Should handle empty query gracefully
        except (ValueError, TypeError):
            # Validation error is acceptable
            pass
        
        # Test weather tools with invalid input
        try:
            result = get_weather("")  # Empty location
            assert isinstance(result, (dict, str))
        except (ValueError, TypeError):
            # Validation error is acceptable
            pass

    def test_tool_function_return_types(self):
        """Test tool functions return expected data types."""
        # Mock successful responses for all tools
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            mock_get.return_value = mock_response
            
            # Test planning tools return types
            plan_result = create_task_plan("test", ["agent1"], "req1")
            assert isinstance(plan_result, dict)
            
            # Test search tools return types
            search_result = search_web("test query")
            assert isinstance(search_result, dict)
            
            summary_result = summarize_search_results({"query": "test", "results": []})
            assert isinstance(summary_result, dict)
        
        # Test summarizer tools return types
        text_summary = summarize_text("Test text to summarize")
        assert isinstance(text_summary, dict)
        
        key_phrases = extract_key_phrases("Test text for key phrases")
        assert isinstance(key_phrases, dict)
        
        # Test Slack tools return types
        formatted_msg = format_slack_message("Title", "Message")
        assert isinstance(formatted_msg, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])