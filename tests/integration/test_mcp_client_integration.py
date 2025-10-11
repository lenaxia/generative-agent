"""
Integration tests for the MCP client functionality.

This test suite verifies the integration of the MCPClientManager with the
overall system, testing actual tool discovery and execution with mocked MCP servers.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_provider.mcp_client import MCPClientManager

# Skip tests if MCP dependencies not available
pytest.importorskip("mcp", reason="MCP dependencies not installed, skipping tests")

# Create a mock MCP server script for testing
MOCK_MCP_SERVER = """
import sys
import json
import time

def read_message():
    length_str = sys.stdin.readline().strip()
    if not length_str:
        return None
    length = int(length_str)
    message = sys.stdin.read(length)
    return json.loads(message)

def write_message(message):
    serialized = json.dumps(message)
    sys.stdout.write(f"{len(serialized)}\\n")
    sys.stdout.write(serialized)
    sys.stdout.flush()

# Mock tools data
TOOLS = [
    {
        "name": "mock_weather",
        "description": "Mock weather forecast tool",
        "parameters": {
            "location": {"type": "string", "description": "Location name"}
        },
        "returns": {"type": "object", "description": "Weather forecast"}
    },
    {
        "name": "mock_search",
        "description": "Mock search tool",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        },
        "returns": {"type": "array", "description": "Search results"}
    }
]

# Main server loop
while True:
    try:
        message = read_message()
        if not message:
            break

        msg_type = message.get("type")
        msg_id = message.get("id")

        if msg_type == "list_tools":
            write_message({"id": msg_id, "tools": TOOLS})
        elif msg_type == "call_tool":
            tool_name = message.get("name")
            parameters = message.get("parameters", {})

            # Mock tool execution
            if tool_name == "mock_weather":
                location = parameters.get("location", "Unknown")
                result = {
                    "location": location,
                    "temperature": 72,
                    "conditions": "Sunny",
                    "forecast": ["Sunny", "Cloudy", "Rain"]
                }
            elif tool_name == "mock_search":
                query = parameters.get("query", "")
                result = [
                    {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
                    {"title": f"Result 2 for {query}", "url": "https://example.com/2"}
                ]
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            write_message({"id": msg_id, "result": result})
        else:
            write_message({"id": msg_id, "error": f"Unknown message type: {msg_type}"})
    except Exception as e:
        # Write error and exit on serious problems
        try:
            write_message({"error": str(e)})
        except:
            pass
        break
"""


@pytest.fixture
def mock_mcp_server():
    """Create a temporary mock MCP server script and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(MOCK_MCP_SERVER)
        server_path = f.name

    yield server_path

    # Clean up after test
    if os.path.exists(server_path):
        os.unlink(server_path)


@pytest.mark.integration
class TestMCPClientIntegration:
    """Integration tests for MCPClientManager with mock MCP servers."""

    def test_register_real_mcp_server(self, mock_mcp_server):
        """Test registering a real MCP server process."""
        # Skip if Python executable not found
        python_executable = sys.executable if "sys" in globals() else "python"

        manager = MCPClientManager()

        try:
            # Register the mock server
            result = manager.register_server(
                name="mock-server",
                command=python_executable,
                args=[mock_mcp_server],
                description="Mock MCP Server",
            )

            # Verify registration
            assert result is True
            assert "mock-server" in manager.clients
            assert (
                len(manager.available_tools) >= 2
            )  # Should have at least our two mock tools

            # Verify tool details
            tool_names = [tool["name"] for tool in manager.available_tools]
            assert "mock_weather" in tool_names
            assert "mock_search" in tool_names

            # Verify server configuration
            assert "mock-server" in manager.server_configs
            assert manager.server_configs["mock-server"].command == python_executable

        finally:
            # Clean up
            manager.cleanup()

    def test_execute_tool_on_real_server(self, mock_mcp_server):
        """Test executing a tool on a real MCP server process."""
        # Skip if Python executable not found
        python_executable = sys.executable if "sys" in globals() else "python"

        manager = MCPClientManager()

        try:
            # Register the mock server
            result = manager.register_server(
                name="mock-server",
                command=python_executable,
                args=[mock_mcp_server],
                description="Mock MCP Server",
            )
            assert result is True

            # Execute the mock_weather tool
            weather_result = manager.execute_tool(
                "mock_weather", {"location": "Seattle"}
            )

            # Verify result
            assert weather_result["location"] == "Seattle"
            assert "temperature" in weather_result
            assert "conditions" in weather_result
            assert "forecast" in weather_result
            assert len(weather_result["forecast"]) == 3

            # Execute the mock_search tool
            search_result = manager.execute_tool("mock_search", {"query": "test query"})

            # Verify result
            assert len(search_result) == 2
            assert search_result[0]["title"] == "Result 1 for test query"
            assert search_result[1]["url"] == "https://example.com/2"

        finally:
            # Clean up
            manager.cleanup()

    def test_error_handling_with_real_server(self, mock_mcp_server):
        """Test error handling with a real MCP server process."""
        # Skip if Python executable not found
        python_executable = sys.executable if "sys" in globals() else "python"

        manager = MCPClientManager()

        try:
            # Register the mock server
            result = manager.register_server(
                name="mock-server",
                command=python_executable,
                args=[mock_mcp_server],
                description="Mock MCP Server",
            )
            assert result is True

            # Try to execute a non-existent tool
            with pytest.raises(ValueError, match="Tool 'non_existent_tool' not found"):
                manager.execute_tool("non_existent_tool", {"param": "value"})

        finally:
            # Clean up
            manager.cleanup()

    @pytest.mark.parametrize(
        "config_file",
        [
            {
                "mcp_servers": {
                    "mock-server": {
                        "command": "python",
                        "args": ["mock_server.py"],
                        "description": "Mock Server",
                    }
                }
            },
            # Empty config
            {},
            # Invalid config (missing command)
            {"mcp_servers": {"invalid-server": {"args": ["mock_server.py"]}}},
        ],
    )
    def test_load_servers_from_config(self, config_file, monkeypatch):
        """Test loading servers from configuration."""
        manager = MCPClientManager()

        # Mock register_server to avoid actual server process
        mock_register = Mock(return_value=True)
        monkeypatch.setattr(manager, "register_server", mock_register)

        # Load from config
        manager.load_servers_from_config(config_file)

        # Check registration calls
        if "mcp_servers" in config_file and config_file["mcp_servers"]:
            valid_servers = [
                name
                for name, cfg in config_file["mcp_servers"].items()
                if "command" in cfg
            ]
            assert mock_register.call_count == len(valid_servers)
        else:
            assert mock_register.call_count == 0
