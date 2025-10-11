"""
Unit tests for the MCP client module.

This test suite covers the MCPClientManager class in the llm_provider/mcp_client.py module,
focusing on both happy paths and error scenarios.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from llm_provider.mcp_client import MCPClientManager, MCPServerConfig

# Skip tests if MCP dependencies not available
pytest.importorskip("mcp", reason="MCP dependencies not installed, skipping tests")


class TestMCPServerConfig:
    """Tests for the MCPServerConfig dataclass."""

    def test_valid_config_creation(self):
        """Test creating a valid server config."""
        config = MCPServerConfig(
            name="test-server",
            command="test-command",
            args=["--arg1", "--arg2"],
            description="Test description",
        )

        assert config.name == "test-server"
        assert config.command == "test-command"
        assert config.args == ["--arg1", "--arg2"]
        assert config.description == "Test description"

    def test_missing_required_fields(self):
        """Test validation for missing required fields."""
        with pytest.raises(
            ValueError, match="MCP server name and command are required"
        ):
            MCPServerConfig(name="", command="command", args=[])

        with pytest.raises(
            ValueError, match="MCP server name and command are required"
        ):
            MCPServerConfig(name="name", command="", args=[])

    def test_invalid_args_type(self):
        """Test validation for args not being a list."""
        with pytest.raises(TypeError, match="MCP server args must be a list"):
            MCPServerConfig(name="test", command="cmd", args="not-a-list")


@patch("llm_provider.mcp_client.MCPClient")
@patch("llm_provider.mcp_client.stdio_client")
@patch("llm_provider.mcp_client.StdioServerParameters")
class TestMCPClientManager:
    """Tests for the MCPClientManager class."""

    def test_initialization(self, mock_params, mock_stdio, mock_client):
        """Test initializing the client manager."""
        manager = MCPClientManager()

        assert manager.clients == {}
        assert manager.available_tools == []
        assert manager.server_configs == {}

    def test_register_server_success(self, mock_params, mock_stdio, mock_client):
        """Test successful server registration."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.__enter__.return_value.list_tools_sync.return_value = [
            {"name": "tool1", "description": "Test tool"}
        ]

        # Create manager and register server
        manager = MCPClientManager()
        result = manager.register_server(
            name="test-server",
            command="test-cmd",
            args=["--arg1"],
            description="Test server",
        )

        # Assertions
        assert result is True
        assert "test-server" in manager.clients
        assert len(manager.available_tools) == 1
        assert manager.available_tools[0]["name"] == "tool1"
        assert manager.available_tools[0]["server"] == "test-server"
        mock_params.assert_called_once_with(command="test-cmd", args=["--arg1"])

    def test_register_server_failure(self, mock_params, mock_stdio, mock_client):
        """Test server registration failure."""
        # Setup mocks to throw an exception
        mock_client.return_value = MagicMock()
        mock_client.return_value.__enter__.side_effect = Exception("Connection failed")

        # Create manager and attempt to register server
        manager = MCPClientManager()
        result = manager.register_server("test-server", "test-cmd", [])

        # Assertions
        assert result is False
        assert "test-server" not in manager.clients
        assert len(manager.available_tools) == 0

    def test_get_tools_for_role_with_role_registry(
        self, mock_params, mock_stdio, mock_client
    ):
        """Test getting tools for a role using role registry."""
        # Setup
        manager = MCPClientManager()
        manager.available_tools = [
            {"name": "weather_tool", "description": "Get weather", "server": "weather"},
            {"name": "search_tool", "description": "Search web", "server": "search"},
        ]

        # Mock role registry
        mock_registry = Mock()
        mock_role = Mock()
        mock_role.config = {"tools": {"mcp_keywords": ["weather"]}}
        mock_registry.get_role.return_value = mock_role

        # Patch the role registry import and global registry getter
        with patch.dict(
            sys.modules,
            {
                "llm_provider.role_registry": Mock(),
            },
        ):
            sys.modules["llm_provider.role_registry"].RoleRegistry = Mock()
            sys.modules[
                "llm_provider.role_registry"
            ].RoleRegistry.get_global_registry = Mock(return_value=mock_registry)

            # Execute
            tools = manager.get_tools_for_role("weather")

            # Assert
            assert len(tools) == 1
            assert tools[0]["name"] == "weather_tool"
            mock_registry.get_role.assert_called_once_with("weather")

    def test_get_tools_for_role_fallback(self, mock_params, mock_stdio, mock_client):
        """Test fallback when role registry not available."""
        # Setup
        manager = MCPClientManager()
        manager.available_tools = [
            {
                "name": "weather_tool",
                "description": "weather forecast",
                "server": "weather",
            },
            {"name": "search_tool", "description": "search engine", "server": "search"},
        ]

        # Patch role registry to raise exception
        with patch.dict(
            sys.modules,
            {
                "llm_provider.role_registry": Mock(),
            },
        ):
            sys.modules["llm_provider.role_registry"].RoleRegistry = Mock()
            sys.modules[
                "llm_provider.role_registry"
            ].RoleRegistry.get_global_registry.side_effect = Exception("Not available")

            # Execute - should use role name as keyword
            tools = manager.get_tools_for_role("weather")

            # Assert - should still find tools with "weather" in name or description
            assert len(tools) == 1
            assert tools[0]["name"] == "weather_tool"

    def test_execute_tool_success(self, mock_params, mock_stdio, mock_client):
        """Test successful tool execution."""
        # Setup
        manager = MCPClientManager()
        mock_client_instance = Mock()
        mock_client_instance.call_tool.return_value = {"result": "success"}
        manager.clients = {"test-server": mock_client_instance}
        manager.available_tools = [{"name": "test_tool", "server": "test-server"}]

        # Execute
        result = manager.execute_tool("test_tool", {"param": "value"})

        # Assert
        assert result == {"result": "success"}
        mock_client_instance.call_tool.assert_called_once_with(
            "test_tool", {"param": "value"}
        )

    def test_execute_tool_not_found(self, mock_params, mock_stdio, mock_client):
        """Test error when tool not found."""
        # Setup
        manager = MCPClientManager()
        manager.available_tools = [{"name": "other_tool", "server": "test-server"}]

        # Execute and assert
        with pytest.raises(ValueError, match="Tool 'test_tool' not found"):
            manager.execute_tool("test_tool", {})

    def test_execute_tool_server_not_available(
        self, mock_params, mock_stdio, mock_client
    ):
        """Test error when server not available."""
        # Setup
        manager = MCPClientManager()
        manager.available_tools = [{"name": "test_tool", "server": "test-server"}]
        # Note: Not adding the server to clients

        # Execute and assert
        with pytest.raises(ValueError, match="MCP server 'test-server' not available"):
            manager.execute_tool("test_tool", {})

    def test_execute_tool_execution_error(self, mock_params, mock_stdio, mock_client):
        """Test error during tool execution."""
        # Setup
        manager = MCPClientManager()
        mock_client_instance = Mock()
        mock_client_instance.call_tool.side_effect = Exception("Execution failed")
        manager.clients = {"test-server": mock_client_instance}
        manager.available_tools = [{"name": "test_tool", "server": "test-server"}]

        # Execute and assert
        with pytest.raises(Exception, match="Execution failed"):
            manager.execute_tool("test_tool", {})

    def test_load_servers_from_config(self, mock_params, mock_stdio, mock_client):
        """Test loading servers from configuration."""
        # Setup
        manager = MCPClientManager()
        manager.register_server = Mock(return_value=True)  # Mock registration

        config_data = {
            "mcp_servers": {
                "server1": {
                    "command": "cmd1",
                    "args": ["--arg1"],
                    "description": "Server 1",
                },
                "server2": {
                    "command": "cmd2",
                    "args": ["--arg2"],
                    "description": "Server 2",
                },
                "invalid_server": {
                    "args": ["--arg3"],  # Missing command
                    "description": "Invalid Server",
                },
            }
        }

        # Execute
        manager.load_servers_from_config(config_data)

        # Assert
        assert manager.register_server.call_count == 2
        # First call - server1
        manager.register_server.assert_any_call(
            "server1", "cmd1", ["--arg1"], "Server 1"
        )
        # Second call - server2
        manager.register_server.assert_any_call(
            "server2", "cmd2", ["--arg2"], "Server 2"
        )

    def test_get_server_status(self, mock_params, mock_stdio, mock_client):
        """Test getting server status."""
        # Setup
        manager = MCPClientManager()
        manager.clients = {"server1": Mock(), "server2": Mock()}
        manager.available_tools = [{"name": "tool1"}, {"name": "tool2"}]
        manager.server_configs = {
            "server1": MCPServerConfig(
                name="server1", command="cmd1", args=["--arg1"], description="Server 1"
            )
        }

        # Execute
        status = manager.get_server_status()

        # Assert
        assert status["mcp_available"] is True  # Assuming MCP is available in test
        assert set(status["registered_servers"]) == {"server1", "server2"}
        assert status["total_tools"] == 2
        assert "server1" in status["server_configs"]
        assert status["server_configs"]["server1"]["command"] == "cmd1"

    def test_cleanup(self, mock_params, mock_stdio, mock_client):
        """Test cleanup of MCP clients."""
        # Setup
        manager = MCPClientManager()
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_client2.__exit__.side_effect = Exception("Cleanup error")
        manager.clients = {"server1": mock_client1, "server2": mock_client2}
        manager.available_tools = [{"name": "tool1"}]
        manager.server_configs = {"server1": Mock()}

        # Execute
        manager.cleanup()

        # Assert
        mock_client1.__exit__.assert_called_once_with(None, None, None)
        mock_client2.__exit__.assert_called_once_with(None, None, None)
        assert manager.clients == {}
        assert manager.available_tools == []
        assert manager.server_configs == {}

    def test_create_mcp_manager_with_defaults(
        self, mock_params, mock_stdio, mock_client
    ):
        """Test creating a manager with defaults."""
        manager = MCPClientManager.create_mcp_manager_with_defaults()
        assert isinstance(manager, MCPClientManager)
