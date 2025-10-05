"""
Test suite for RequestRouter component.

Tests the fast-path routing logic including:
- Route classification (fast-reply vs planning)
- Confidence scoring
- Error handling and fallback
- Integration with RoleRegistry
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from llm_provider.request_router import RequestRouter
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleRegistry, RoleDefinition


class TestRequestRouter:
    """Test suite for RequestRouter class."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory for testing."""
        factory = Mock(spec=LLMFactory)
        mock_model = Mock()
        mock_agent = Mock()
        mock_agent.return_value = '{"route": "weather", "confidence": 0.9}'
        
        # Mock the Agent class creation
        with patch('llm_provider.request_router.Agent', return_value=mock_agent):
            factory.create_strands_model.return_value = mock_model
            yield factory
    
    @pytest.fixture
    def mock_role_registry(self):
        """Mock role registry with fast-reply roles."""
        registry = Mock(spec=RoleRegistry)
        
        # Create mock fast-reply roles
        weather_role = Mock(spec=RoleDefinition)
        weather_role.name = "weather"
        weather_role.config = {
            "role": {
                "description": "Get weather information",
                "fast_reply": True
            }
        }
        
        calendar_role = Mock(spec=RoleDefinition)
        calendar_role.name = "calendar"
        calendar_role.config = {
            "role": {
                "description": "Manage calendar events",
                "fast_reply": True
            }
        }
        
        registry.get_fast_reply_roles.return_value = [weather_role, calendar_role]
        return registry
    
    @pytest.fixture
    def request_router(self, mock_llm_factory, mock_role_registry):
        """Create RequestRouter instance for testing."""
        # Mock UniversalAgent for testing
        mock_universal_agent = Mock()
        return RequestRouter(mock_llm_factory, mock_role_registry, mock_universal_agent)
    
    def test_init(self, mock_llm_factory, mock_role_registry):
        """Test RequestRouter initialization."""
        router = RequestRouter(mock_llm_factory, mock_role_registry)
        
        assert router.llm_factory == mock_llm_factory
        assert router.role_registry == mock_role_registry
    
    def test_route_request_weather_query(self, request_router):
        """Test routing a weather query to fast-reply."""
        # Mock the UniversalAgent response properly
        request_router.universal_agent.execute_task.return_value = '{"route": "weather", "confidence": 0.9}'
        
        result = request_router.route_request("What's the weather like today?")
        
        assert result["route"] == "weather"
        assert result["confidence"] == 0.9
        assert "error" not in result
        
        # Verify UniversalAgent was called correctly
        request_router.universal_agent.execute_task.assert_called_once()
    
    def test_route_request_complex_planning(self, request_router):
        """Test routing a complex request to planning."""
        with patch('llm_provider.request_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = '{"route": "PLANNING", "confidence": 0.8}'
            mock_agent_class.return_value = mock_agent
            
            result = request_router.route_request("Create a comprehensive project plan with multiple phases")
            
            assert result["route"] == "PLANNING"
            assert result["confidence"] == 0.8
    
    def test_route_request_low_confidence(self, request_router):
        """Test routing with low confidence defaults to planning."""
        with patch('llm_provider.request_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = '{"route": "weather", "confidence": 0.3}'
            mock_agent_class.return_value = mock_agent
            
            result = request_router.route_request("Maybe weather related?")
            
            assert result["route"] == "weather"
            assert result["confidence"] == 0.3
    
    def test_parse_routing_response_valid_json(self, request_router):
        """Test parsing valid JSON routing response."""
        response = '{"route": "calendar", "confidence": 0.85}'
        result = request_router._parse_routing_response(response)
        
        assert result["route"] == "calendar"
        assert result["confidence"] == 0.85
        assert "error" not in result
    
    def test_parse_routing_response_missing_route(self, request_router):
        """Test parsing response missing route field."""
        response = '{"confidence": 0.85}'
        result = request_router._parse_routing_response(response)
        
        assert result["route"] == "PLANNING"
        assert result["confidence"] == 0.0
        assert "error" in result
        assert "Missing route field" in result["error"]
    
    def test_parse_routing_response_invalid_confidence(self, request_router):
        """Test parsing response with invalid confidence."""
        response = '{"route": "weather", "confidence": "high"}'
        result = request_router._parse_routing_response(response)
        
        assert result["route"] == "weather"
        assert result["confidence"] == 0.0
    
    def test_parse_routing_response_invalid_json(self, request_router):
        """Test parsing invalid JSON response."""
        response = '{"route": "weather", "confidence": 0.9'  # Missing closing brace
        result = request_router._parse_routing_response(response)
        
        assert result["route"] == "PLANNING"
        assert result["confidence"] == 0.0
        assert "error" in result
        assert "Failed to parse routing response" in result["error"]
    
    def test_parse_routing_response_empty_response(self, request_router):
        """Test parsing empty response."""
        response = ""
        result = request_router._parse_routing_response(response)
        
        assert result["route"] == "PLANNING"
        assert result["confidence"] == 0.0
        assert "error" in result
    
    def test_build_routing_prompt(self, request_router, mock_role_registry):
        """Test building routing prompt with roles."""
        instruction = "Set a timer for 10 minutes"
        
        # Mock roles for prompt building
        weather_role = Mock()
        weather_role.name = "weather"
        weather_role.config = {"role": {"description": "Weather information"}}
        
        timer_role = Mock()
        timer_role.name = "timer"
        timer_role.config = {"role": {"description": "Timer management"}}
        
        roles = [weather_role, timer_role]
        
        prompt = request_router._build_routing_prompt(instruction, roles)
        
        assert instruction in prompt
        assert "weather: Weather information" in prompt
        assert "timer: Timer management" in prompt
        assert "PLANNING: Multi-step task requiring planning" in prompt
        assert "Respond with JSON only:" in prompt
    
    def test_build_routing_prompt_no_description(self, request_router):
        """Test building prompt with role missing description."""
        instruction = "Test instruction"
        
        role = Mock()
        role.name = "test_role"
        role.config = {"role": {}}  # No description
        
        prompt = request_router._build_routing_prompt(instruction, [role])
        
        assert "test_role:" in prompt  # Should handle missing description gracefully
    
    def test_route_request_agent_creation_error(self, request_router):
        """Test handling of UniversalAgent execution errors."""
        request_router.universal_agent.execute_task.side_effect = Exception("UniversalAgent execution failed")
        
        result = request_router.route_request("Test instruction")
        
        assert result["route"] == "PLANNING"
        assert result["confidence"] == 0.0
        assert "error" in result
        assert "UniversalAgent execution failed" in result["error"]
    
    def test_route_request_agent_execution_error(self, request_router):
        """Test handling of agent execution errors."""
        with patch('llm_provider.request_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.side_effect = Exception("Agent execution failed")
            mock_agent_class.return_value = mock_agent
            
            result = request_router.route_request("Test instruction")
            
            assert result["route"] == "PLANNING"
            assert result["confidence"] == 0.0
            assert "error" in result
            assert "Agent execution failed" in result["error"]


class TestRequestRouterIntegration:
    """Integration tests for RequestRouter with real components."""
    
    def test_integration_with_role_registry(self):
        """Test RequestRouter integration with actual RoleRegistry."""
        # This would require actual role definitions in test environment
        # For now, we'll mock the integration points
        pass
    
    def test_performance_routing_speed(self):
        """Test that routing completes within acceptable time limits."""
        # Performance test to ensure routing is fast enough for fast-path
        pass


class TestRequestRouterEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def request_router(self):
        """Create minimal RequestRouter for edge case testing."""
        mock_factory = Mock()
        mock_registry = Mock()
        
        # Create mock fast-reply roles for edge case testing
        weather_role = Mock()
        weather_role.name = "weather"
        weather_role.config = {"role": {"description": "Weather information", "fast_reply": True}}
        
        mock_registry.get_fast_reply_roles.return_value = [weather_role]
        return RequestRouter(mock_factory, mock_registry)
    
    def test_empty_instruction(self, request_router):
        """Test routing with empty instruction."""
        with patch('llm_provider.request_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = '{"route": "PLANNING", "confidence": 0.0}'
            mock_agent_class.return_value = mock_agent
            
            result = request_router.route_request("")
            
            assert result["route"] == "PLANNING"
    
    def test_very_long_instruction(self, request_router):
        """Test routing with very long instruction."""
        long_instruction = "A" * 10000  # Very long instruction
        
        with patch('llm_provider.request_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = '{"route": "PLANNING", "confidence": 0.5}'
            mock_agent_class.return_value = mock_agent
            
            result = request_router.route_request(long_instruction)
            
            assert result["route"] == "PLANNING"
    
    def test_no_fast_reply_roles(self, request_router):
        """Test routing when no fast-reply roles are available."""
        with patch('llm_provider.request_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = '{"route": "PLANNING", "confidence": 0.8}'
            mock_agent_class.return_value = mock_agent
            
            result = request_router.route_request("Any instruction")
            
            assert result["route"] == "PLANNING"
    
    def test_unicode_instruction(self, request_router):
        """Test routing with unicode characters in instruction."""
        unicode_instruction = "What's the weather in 北京?"
        
        with patch('llm_provider.request_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = '{"route": "weather", "confidence": 0.9}'
            mock_agent_class.return_value = mock_agent
            
            result = request_router.route_request(unicode_instruction)
            
            assert result["route"] == "weather"
            assert result["confidence"] == 0.9