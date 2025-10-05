"""
Unit tests for the Dynamic Role System.

Tests the RoleRegistry, role loading, and integration with UniversalAgent.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from llm_provider.role_registry import RoleRegistry, RoleDefinition
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType


class TestDynamicRoleSystem:
    """Test suite for the dynamic role system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test roles
        self.temp_dir = tempfile.mkdtemp()
        self.roles_dir = Path(self.temp_dir) / "roles"
        self.roles_dir.mkdir()
        
        # Create shared_tools directory
        shared_tools_dir = self.roles_dir / "shared_tools"
        shared_tools_dir.mkdir()
        
        # Create a simple shared tool
        (shared_tools_dir / "__init__.py").write_text("")
        (shared_tools_dir / "test_tool.py").write_text("""
from strands import tool

@tool
def test_shared_tool(input_text: str) -> str:
    return f"Processed: {input_text}"
""")
        
        # Create test role
        test_role_dir = self.roles_dir / "test_role"
        test_role_dir.mkdir()
        
        # Create role definition
        (test_role_dir / "definition.yaml").write_text("""
role:
  name: "test_role"
  version: "1.0.0"
  description: "Test role for unit testing"
  when_to_use: "Use for testing purposes"

model_config:
  temperature: 0.5
  max_tokens: 2048

prompts:
  system: "You are a test role agent."

tools:
  automatic: false
  shared:
    - "test_shared_tool"

capabilities:
  max_iterations: 5
  timeout_seconds: 120

logging:
  level: "DEBUG"
  include_tool_calls: true
""")
        
        # Create custom tools for the role
        (test_role_dir / "tools.py").write_text("""
from strands import tool

@tool
def test_custom_tool(data: str) -> dict:
    return {"result": f"Custom processing: {data}"}

@tool
def another_custom_tool(value: int) -> int:
    return value * 2
""")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_role_registry_initialization(self):
        """Test RoleRegistry initialization and role discovery."""
        registry = RoleRegistry(str(self.roles_dir))
        
        # Should discover the test role
        assert "test_role" in registry.roles
        assert len(registry.roles) == 1
        
        # Check role definition
        role_def = registry.get_role("test_role")
        assert role_def is not None
        assert role_def.name == "test_role"
        assert role_def.config['role']['description'] == "Test role for unit testing"
    
    def test_shared_tools_loading(self):
        """Test loading of shared tools."""
        registry = RoleRegistry(str(self.roles_dir))
        
        # Should have loaded shared tools
        assert len(registry.shared_tools) > 0
        assert "test_shared_tool" in registry.shared_tools
        
        # Test getting shared tool
        shared_tool = registry.get_shared_tool("test_shared_tool")
        assert shared_tool is not None
        assert callable(shared_tool)
    
    def test_custom_tools_loading(self):
        """Test loading of custom tools from role's tools.py."""
        registry = RoleRegistry(str(self.roles_dir))
        role_def = registry.get_role("test_role")
        
        # Should have loaded custom tools
        assert len(role_def.custom_tools) == 2
        
        # Check tool names
        tool_names = [getattr(tool, '_tool_name', tool.__name__) for tool in role_def.custom_tools]
        assert "test_custom_tool" in tool_names
        assert "another_custom_tool" in tool_names
    
    def test_role_summaries(self):
        """Test getting role summaries for planning."""
        registry = RoleRegistry(str(self.roles_dir))
        summaries = registry.get_role_summaries()
        
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary['name'] == "test_role"
        assert summary['description'] == "Test role for unit testing"
        assert summary['when_to_use'] == "Use for testing purposes"
    
    def test_role_validation(self):
        """Test role definition validation."""
        registry = RoleRegistry(str(self.roles_dir))
        
        # Valid role should pass validation
        validation = registry.validate_role("test_role")
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
        
        # Non-existent role should fail
        validation = registry.validate_role("non_existent")
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
    
    def test_registry_statistics(self):
        """Test registry statistics."""
        registry = RoleRegistry(str(self.roles_dir))
        stats = registry.get_statistics()
        
        assert stats['total_roles'] == 1
        assert stats['total_shared_tools'] >= 1
        assert stats['total_custom_tools'] == 2
        assert stats['roles_with_custom_tools'] == 1
    
    @patch('llm_provider.role_registry.RoleRegistry.get_global_registry')
    def test_universal_agent_with_dynamic_roles(self, mock_registry):
        """Test UniversalAgent integration with dynamic roles."""
        # Mock the role registry
        mock_role_def = Mock(spec=RoleDefinition)
        mock_role_def.name = "test_role"
        mock_role_def.config = {
            'role': {'name': 'test_role'},
            'prompts': {'system': 'Test system prompt'},
            'model_config': {'temperature': 0.5},
            'tools': {'automatic': False, 'shared': []}
        }
        mock_role_def.custom_tools = []
        
        mock_registry_instance = Mock()
        mock_registry_instance.get_role.return_value = mock_role_def
        mock_registry_instance.get_shared_tool.return_value = None
        mock_registry.return_value = mock_registry_instance
        
        # Create mock LLM factory
        mock_llm_factory = Mock()
        mock_model = Mock()
        mock_llm_factory.create_strands_model.return_value = mock_model
        
        # Create UniversalAgent with role registry
        universal_agent = UniversalAgent(mock_llm_factory, role_registry=mock_registry_instance)
        
        # Test role assumption with comprehensive mocking to avoid LLM calls
        with patch.object(universal_agent, '_create_strands_model') as mock_create_model, \
             patch('strands.Agent') as mock_agent_class:
            
            mock_create_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            agent = universal_agent.assume_role("test_role")
            
            # Verify role registry was called
            mock_registry_instance.get_role.assert_called_once_with("test_role")
            
            # Verify agent was created
            assert agent is not None
            assert universal_agent.current_role == "test_role"
    
    def test_role_not_found_fallback(self):
        """Test fallback behavior when role is not found."""
        registry = RoleRegistry(str(self.roles_dir))
        
        # Mock LLM factory
        mock_llm_factory = Mock()
        mock_model = Mock()
        mock_llm_factory.create_chat_model.return_value = mock_model
        
        universal_agent = UniversalAgent(mock_llm_factory, role_registry=registry)
        
        with patch('llm_provider.universal_agent.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Try to assume non-existent role
            agent = universal_agent.assume_role("non_existent_role")
            
            # Should fall back to default agent
            assert agent == mock_agent
            # Current role should be the dynamically generated role name
            assert universal_agent.current_role == "dynamic_non_existent_role"
    
    def test_role_refresh(self):
        """Test role registry refresh functionality."""
        registry = RoleRegistry(str(self.roles_dir))
        
        # Initially should have 1 role
        assert len(registry.roles) == 1
        
        # Create another role
        new_role_dir = self.roles_dir / "new_test_role"
        new_role_dir.mkdir()
        (new_role_dir / "definition.yaml").write_text("""
role:
  name: "new_test_role"
  description: "Another test role"
  when_to_use: "For additional testing"

prompts:
  system: "You are another test role."

tools:
  automatic: false

capabilities:
  max_iterations: 3
  timeout_seconds: 60

logging:
  level: "INFO"
  include_tool_calls: false
""")
        
        # Refresh registry
        registry.refresh()
        
        # Should now have 2 roles
        assert len(registry.roles) == 2
        assert "new_test_role" in registry.roles
    
    def test_role_list_functionality(self):
        """Test role listing functionality."""
        registry = RoleRegistry(str(self.roles_dir))
        roles_list = registry.list_roles()
        
        assert len(roles_list) == 1
        role_info = roles_list[0]
        assert role_info['name'] == "test_role"
        assert role_info['description'] == "Test role for unit testing"
        assert role_info['version'] == "1.0.0"
        assert role_info['custom_tool_count'] == 2
        assert role_info['has_automatic_tools'] is False
        assert "test_shared_tool" in role_info['shared_tools']