"""
Tests for MCP (Model Context Protocol) server integration with Universal Agent.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_provider.mcp_client import MCPClientManager
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType


class TestMCPClientManager:
    """Test MCP client management functionality."""
    
    def test_init_mcp_client_manager(self):
        """Test MCPClientManager initialization."""
        manager = MCPClientManager()
        assert manager.clients == {}
        assert manager.available_tools == []
    
    def test_register_mcp_server_aws_docs(self):
        """Test registering AWS documentation MCP server."""
        manager = MCPClientManager()
        
        # Mock MCP availability
        with patch('llm_provider.mcp_client.MCP_AVAILABLE', True), \
             patch('llm_provider.mcp_client.MCPClient') as mock_mcp_client:
            
            mock_client = Mock()
            mock_client.list_tools_sync.return_value = [
                {'name': 'search_aws_docs', 'description': 'Search AWS documentation'},
                {'name': 'get_aws_service_info', 'description': 'Get AWS service information'}
            ]
            mock_mcp_client.return_value = mock_client
            
            # Mock context manager behavior
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            
            # Register AWS docs server
            result = manager.register_server(
                name="aws_docs",
                command="uvx",
                args=["awslabs.aws-documentation-mcp-server@latest"]
            )
            
            assert result is True
            assert "aws_docs" in manager.clients
            assert len(manager.available_tools) == 2
            assert any(tool['name'] == 'search_aws_docs' for tool in manager.available_tools)
    
    def test_register_mcp_server_web_search(self):
        """Test registering web search MCP server."""
        manager = MCPClientManager()
        
        with patch('llm_provider.mcp_client.MCP_AVAILABLE', True), \
             patch('llm_provider.mcp_client.MCPClient') as mock_mcp_client:
            
            mock_client = Mock()
            mock_client.list_tools_sync.return_value = [
                {'name': 'web_search', 'description': 'Search the web'},
                {'name': 'get_webpage_content', 'description': 'Get webpage content'}
            ]
            mock_mcp_client.return_value = mock_client
            
            # Mock context manager behavior
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            
            result = manager.register_server(
                name="web_search",
                command="npx",
                args=["@modelcontextprotocol/server-web-search"]
            )
            
            assert result is True
            assert "web_search" in manager.clients
            assert len(manager.available_tools) == 2
            assert any(tool['name'] == 'web_search' for tool in manager.available_tools)
    
    def test_get_tools_for_role_planning(self):
        """Test getting MCP tools for planning role."""
        manager = MCPClientManager()
        
        # Mock tools from different servers
        manager.available_tools = [
            {'name': 'search_aws_docs', 'description': 'Search AWS documentation', 'server': 'aws_docs'},
            {'name': 'web_search', 'description': 'Search the web', 'server': 'web_search'},
            {'name': 'get_weather', 'description': 'Get weather information', 'server': 'weather'}
        ]
        
        # Planning role should get research and documentation tools
        planning_tools = manager.get_tools_for_role("planning")
        
        assert len(planning_tools) >= 2
        tool_names = [tool['name'] for tool in planning_tools]
        assert 'search_aws_docs' in tool_names
        assert 'web_search' in tool_names
    
    def test_get_tools_for_role_search(self):
        """Test getting MCP tools for search role."""
        manager = MCPClientManager()
        
        manager.available_tools = [
            {'name': 'search_aws_docs', 'description': 'Search AWS documentation', 'server': 'aws_docs'},
            {'name': 'web_search', 'description': 'Search the web', 'server': 'web_search'},
            {'name': 'get_weather', 'description': 'Get weather information', 'server': 'weather'}
        ]
        
        # Search role should primarily get search-related tools
        search_tools = manager.get_tools_for_role("search")
        
        assert len(search_tools) >= 1
        tool_names = [tool['name'] for tool in search_tools]
        assert 'web_search' in tool_names
    
    def test_execute_mcp_tool(self):
        """Test executing an MCP tool through the manager."""
        manager = MCPClientManager()
        
        # Mock client and tool execution
        mock_client = Mock()
        mock_client.call_tool.return_value = {"result": "AWS Bedrock is a managed service..."}
        manager.clients["aws_docs"] = mock_client
        
        # Add the tool to available_tools
        manager.available_tools = [
            {'name': 'search_aws_docs', 'description': 'Search AWS documentation', 'server': 'aws_docs'}
        ]
        
        result = manager.execute_tool("search_aws_docs", {"query": "Amazon Bedrock"})
        
        assert result["result"] == "AWS Bedrock is a managed service..."
        mock_client.call_tool.assert_called_once_with("search_aws_docs", {"query": "Amazon Bedrock"})
    
    def test_cleanup_mcp_clients(self):
        """Test cleaning up MCP clients."""
        manager = MCPClientManager()
        
        # Mock clients with proper context manager behavior
        mock_client1 = Mock()
        mock_client1.__exit__ = Mock(return_value=None)
        mock_client2 = Mock()
        mock_client2.__exit__ = Mock(return_value=None)
        
        manager.clients = {"aws_docs": mock_client1, "web_search": mock_client2}
        
        manager.cleanup()
        
        mock_client1.__exit__.assert_called_once_with(None, None, None)
        mock_client2.__exit__.assert_called_once_with(None, None, None)
        assert manager.clients == {}
        assert manager.available_tools == []


class TestUniversalAgentMCPIntegration:
    """Test Universal Agent integration with MCP servers."""
    
    def test_universal_agent_with_mcp_tools(self):
        """Test Universal Agent using MCP tools."""
        # Mock LLM factory
        mock_factory = Mock(spec=LLMFactory)
        mock_factory.get_framework.return_value = 'strands'
        
        # Mock MCP manager
        mock_mcp_manager = Mock()
        mock_mcp_manager.get_tools_for_role.return_value = [
            {'name': 'search_aws_docs', 'description': 'Search AWS documentation'},
            {'name': 'web_search', 'description': 'Search the web'}
        ]
        
        # Create Universal Agent with MCP integration
        agent = UniversalAgent(mock_factory, mcp_manager=mock_mcp_manager)
        
        # Test role assumption with MCP tools
        with patch.object(agent, '_create_strands_model') as mock_create_model, \
             patch.object(agent, '_get_role_prompt') as mock_get_prompt, \
             patch.object(agent, 'strands_available', True):
            
            mock_create_model.return_value = Mock()
            mock_get_prompt.return_value = "Test prompt"
            
            role_agent = agent.assume_role(
                role="planning",
                llm_type=LLMType.STRONG,
                tools=["search_aws_docs", "web_search"]
            )
            
            assert role_agent is not None
            mock_mcp_manager.get_tools_for_role.assert_called_with("planning")
    
    def test_universal_agent_mcp_tool_execution(self):
        """Test Universal Agent executing MCP tools."""
        mock_factory = Mock(spec=LLMFactory)
        mock_factory.get_framework.return_value = 'strands'
        
        # Mock MCP manager with tool execution
        mock_mcp_manager = Mock()
        mock_mcp_manager.execute_tool.return_value = {
            "result": "Amazon Bedrock provides managed foundation models..."
        }
        
        agent = UniversalAgent(mock_factory, mcp_manager=mock_mcp_manager)
        
        # Execute MCP tool
        result = agent.execute_mcp_tool("search_aws_docs", {"query": "Amazon Bedrock"})
        
        assert "Amazon Bedrock provides managed foundation models" in result["result"]
        mock_mcp_manager.execute_tool.assert_called_once_with(
            "search_aws_docs", 
            {"query": "Amazon Bedrock"}
        )


class TestMCPServerConfiguration:
    """Test MCP server configuration and setup."""
    
    def test_mcp_server_config_validation(self):
        """Test MCP server configuration validation."""
        from llm_provider.mcp_client import MCPServerConfig
        
        # Valid configuration
        config = MCPServerConfig(
            name="aws_docs",
            command="uvx",
            args=["awslabs.aws-documentation-mcp-server@latest"],
            description="AWS documentation server"
        )
        
        assert config.name == "aws_docs"
        assert config.command == "uvx"
        assert len(config.args) == 1
    
    def test_mcp_server_config_invalid(self):
        """Test invalid MCP server configuration."""
        from llm_provider.mcp_client import MCPServerConfig
        
        # Missing required fields should raise validation error
        with pytest.raises((ValueError, TypeError)):
            MCPServerConfig(name="", command="", args=[])
    
    def test_load_mcp_servers_from_config(self):
        """Test loading MCP servers from configuration."""
        config_data = {
            "mcp_servers": {
                "aws_docs": {
                    "command": "uvx",
                    "args": ["awslabs.aws-documentation-mcp-server@latest"],
                    "description": "AWS documentation server"
                },
                "web_search": {
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-web-search"],
                    "description": "Web search server"
                }
            }
        }
        
        manager = MCPClientManager()
        
        with patch.object(manager, 'register_server') as mock_register:
            manager.load_servers_from_config(config_data)
            
            assert mock_register.call_count == 2
            
            # Check AWS docs server registration
            mock_register.assert_any_call(
                "aws_docs",
                "uvx",
                ["awslabs.aws-documentation-mcp-server@latest"],
                "AWS documentation server"
            )
            
            # Check web search server registration
            mock_register.assert_any_call(
                "web_search",
                "npx",
                ["@modelcontextprotocol/server-web-search"],
                "Web search server"
            )


    def test_mcp_unavailable_scenario(self):
        """Test MCP client manager when MCP dependencies are not available."""
        manager = MCPClientManager()
        
        # When MCP is not available, registration should fail gracefully
        result = manager.register_server(
            name="test_server",
            command="uvx",
            args=["test-server"]
        )
        
        assert result is False
        assert len(manager.clients) == 0
        assert len(manager.available_tools) == 0
        
        # Status should reflect MCP unavailability
        status = manager.get_server_status()
        assert status["mcp_available"] is False
        assert status["registered_servers"] == []
        assert status["total_tools"] == 0


if __name__ == "__main__":
    pytest.main([__file__])