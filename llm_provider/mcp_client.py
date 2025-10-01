"""
MCP (Model Context Protocol) client integration for Universal Agent.

This module provides MCP server management and tool integration capabilities
for the Universal Agent system, enabling access to external tool ecosystems.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from strands.tools.mcp import MCPClient
    from mcp import stdio_client, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPClient = None
    stdio_client = None
    StdioServerParameters = None

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str]
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name or not self.command:
            raise ValueError("MCP server name and command are required")
        if not isinstance(self.args, list):
            raise TypeError("MCP server args must be a list")


class MCPClientManager:
    """
    Manages MCP (Model Context Protocol) server connections and tool access.
    
    This class handles:
    - MCP server registration and lifecycle management
    - Tool discovery and role-based tool selection
    - Tool execution through MCP clients
    - Configuration loading and validation
    """
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.clients: Dict[str, Any] = {}
        self.available_tools: List[Dict[str, Any]] = []
        self.server_configs: Dict[str, MCPServerConfig] = {}
        
        if not MCP_AVAILABLE:
            logger.warning("MCP dependencies not available. MCP integration disabled.")
    
    def register_server(self, name: str, command: str, args: List[str], 
                       description: Optional[str] = None) -> bool:
        """
        Register and initialize an MCP server.
        
        Args:
            name: Unique name for the MCP server
            command: Command to start the server (e.g., 'uvx', 'npx')
            args: Arguments for the server command
            description: Optional description of the server
            
        Returns:
            True if server registered successfully, False otherwise
        """
        if not MCP_AVAILABLE:
            logger.error("name=<%s> | MCP dependencies not available", name)
            return False
        
        try:
            # Create server configuration
            config = MCPServerConfig(name, command, args, description)
            self.server_configs[name] = config
            
            # Create MCP client
            client = MCPClient(
                lambda: stdio_client(StdioServerParameters(command=command, args=args))
            )
            
            # Initialize client and get tools
            with client:
                tools = client.list_tools_sync()
                
                # Add server name to each tool for tracking
                for tool in tools:
                    tool['server'] = name
                
                self.available_tools.extend(tools)
                self.clients[name] = client
                
                logger.info(
                    "name=<%s>, tools_count=<%d> | MCP server registered successfully",
                    name, len(tools)
                )
                return True
                
        except Exception as e:
            logger.error(
                "name=<%s>, error=<%s> | Failed to register MCP server",
                name, str(e)
            )
            return False
    
    def get_tools_for_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get MCP tools appropriate for a specific agent role.
        
        Args:
            role: Agent role (planning, search, weather, etc.)
            
        Returns:
            List of tools suitable for the role
        """
        role_tool_mapping = {
            "planning": ["search", "docs", "documentation", "aws", "research"],
            "search": ["search", "web", "query"],
            "weather": ["weather", "forecast", "climate"],
            "summarizer": ["text", "content", "document"],
            "slack": ["slack", "message", "communication"]
        }
        
        role_keywords = role_tool_mapping.get(role, [])
        if not role_keywords:
            return self.available_tools  # Return all tools if role not mapped
        
        # Filter tools based on role keywords
        relevant_tools = []
        for tool in self.available_tools:
            tool_name = tool.get('name', '').lower()
            tool_desc = tool.get('description', '').lower()
            
            # Check if any role keyword matches tool name or description
            if any(keyword in tool_name or keyword in tool_desc 
                   for keyword in role_keywords):
                relevant_tools.append(tool)
        
        logger.debug(
            "role=<%s>, available_tools=<%d>, relevant_tools=<%d> | filtered tools for role",
            role, len(self.available_tools), len(relevant_tools)
        )
        
        return relevant_tools
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or server not available
        """
        # Find which server provides this tool
        tool_server = None
        for tool in self.available_tools:
            if tool.get('name') == tool_name:
                tool_server = tool.get('server')
                break
        
        if not tool_server:
            raise ValueError(f"Tool '{tool_name}' not found in available MCP tools")
        
        if tool_server not in self.clients:
            raise ValueError(f"MCP server '{tool_server}' not available")
        
        try:
            client = self.clients[tool_server]
            result = client.call_tool(tool_name, parameters)
            
            logger.debug(
                "tool_name=<%s>, server=<%s> | MCP tool executed successfully",
                tool_name, tool_server
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "tool_name=<%s>, server=<%s>, error=<%s> | MCP tool execution failed",
                tool_name, tool_server, str(e)
            )
            raise
    
    def load_servers_from_config(self, config_data: Dict[str, Any]) -> None:
        """
        Load and register MCP servers from configuration data.
        
        Args:
            config_data: Configuration dictionary containing MCP server definitions
        """
        mcp_config = config_data.get('mcp_servers', {})
        
        for server_name, server_config in mcp_config.items():
            command = server_config.get('command')
            args = server_config.get('args', [])
            description = server_config.get('description')
            
            if not command:
                logger.warning(
                    "server_name=<%s> | Missing command in MCP server configuration",
                    server_name
                )
                continue
            
            self.register_server(server_name, command, args, description)
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get status information for all registered MCP servers.
        
        Returns:
            Dictionary containing server status information
        """
        return {
            "mcp_available": MCP_AVAILABLE,
            "registered_servers": list(self.clients.keys()),
            "total_tools": len(self.available_tools),
            "server_configs": {
                name: {
                    "command": config.command,
                    "args": config.args,
                    "description": config.description
                }
                for name, config in self.server_configs.items()
            }
        }
    
    def cleanup(self) -> None:
        """Clean up MCP clients and connections."""
        for name, client in self.clients.items():
            try:
                client.__exit__(None, None, None)
                logger.debug("name=<%s> | MCP client cleaned up", name)
            except Exception as e:
                logger.warning(
                    "name=<%s>, error=<%s> | Error cleaning up MCP client",
                    name, str(e)
                )
        
        self.clients.clear()
        self.available_tools.clear()
        self.server_configs.clear()


# Predefined MCP server configurations for common use cases
COMMON_MCP_SERVERS = {
    "aws_docs": MCPServerConfig(
        name="aws_docs",
        command="uvx",
        args=["awslabs.aws-documentation-mcp-server@latest"],
        description="AWS documentation and service information"
    ),
    "web_search": MCPServerConfig(
        name="web_search", 
        command="npx",
        args=["@modelcontextprotocol/server-web-search"],
        description="Web search capabilities"
    ),
    "weather": MCPServerConfig(
        name="weather",
        command="npx", 
        args=["@modelcontextprotocol/server-weather"],
        description="Weather information and forecasts"
    ),
    "filesystem": MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["@modelcontextprotocol/server-filesystem"],
        description="File system operations"
    )
}


def create_mcp_manager_with_defaults() -> MCPClientManager:
    """
    Create an MCP client manager with common servers pre-configured.
    
    Returns:
        MCPClientManager with default servers registered
    """
    manager = MCPClientManager()
    
    if not MCP_AVAILABLE:
        logger.warning("MCP not available, returning empty manager")
        return manager
    
    # Register common servers
    for server_config in COMMON_MCP_SERVERS.values():
        manager.register_server(
            server_config.name,
            server_config.command, 
            server_config.args,
            server_config.description
        )
    
    return manager