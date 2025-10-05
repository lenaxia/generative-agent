import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.mcp_client import MCPClientManager


class TestUniversalAgentMCPIntegration:
    """Test UniversalAgent MCP integration with actual implementation."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value='strands')
        return factory
    
    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = Mock(spec=MCPClientManager)
        manager.get_tools_for_role = Mock(return_value=[])
        manager.get_registered_servers = Mock(return_value=[])
        manager.get_all_tools = Mock(return_value=[])
        manager.get_server_configs = Mock(return_value={})
        return manager
    
    @pytest.fixture
    def universal_agent(self, mock_llm_factory, mock_mcp_manager):
        """Create a UniversalAgent with MCP integration."""
        return UniversalAgent(mock_llm_factory, mcp_manager=mock_mcp_manager)
    
    def test_universal_agent_with_mcp_tools(self, universal_agent):
        """Test Universal Agent with MCP tools integration."""
        # Test that MCP manager is integrated
        assert universal_agent.mcp_manager is not None
        
        # Test role assumption with MCP tools
        with patch.object(universal_agent, '_create_strands_model') as mock_model, \
             patch('llm_provider.universal_agent.Agent') as mock_agent_class:
            
            mock_model.return_value = Mock()
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            agent = universal_agent.assume_role("planning")
            assert agent is not None
    
    def test_universal_agent_mcp_tool_execution(self, universal_agent):
        """Test Universal Agent MCP tool execution."""
        # Test that MCP status can be retrieved
        with patch.object(universal_agent.mcp_manager, 'get_registered_servers', return_value=[]):
            status = universal_agent.get_mcp_status()
            assert isinstance(status, dict)
            assert 'mcp_available' in status


class TestMCPServerConfiguration:
    """Test MCP server configuration scenarios."""
    
    @pytest.fixture
    def mock_llm_factory(self):
        """Create a mock LLM factory."""
        factory = Mock(spec=LLMFactory)
        factory.get_framework = Mock(return_value='strands')
        return factory
    
    def test_mcp_unavailable_scenario(self, mock_llm_factory):
        """Test behavior when MCP is unavailable."""
        # Test with no MCP manager
        universal_agent = UniversalAgent(mock_llm_factory, mcp_manager=None)
        
        # Should still work without MCP
        assert universal_agent.mcp_manager is None
        
        # MCP status should indicate unavailable
        status = universal_agent.get_mcp_status()
        assert status["mcp_available"] is False


if __name__ == "__main__":
    pytest.main([__file__])