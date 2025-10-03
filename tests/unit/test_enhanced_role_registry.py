"""
Unit tests for enhanced RoleRegistry with programmatic role support.

Tests the enhanced RoleRegistry that supports both YAML-based LLM roles
and Python-based programmatic roles for hybrid execution architecture.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional

from llm_provider.role_registry import RoleRegistry, RoleDefinition
from llm_provider.programmatic_role import ProgrammaticRole
from common.task_context import TaskContext


class MockProgrammaticRole(ProgrammaticRole):
    """Mock programmatic role for testing."""
    
    def execute(self, instruction: str, context: Optional[TaskContext] = None) -> Any:
        return {"mock_result": instruction, "context_provided": context is not None}
    
    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        return {"parsed": instruction.split()}


class TestEnhancedRoleRegistry:
    """Test suite for enhanced RoleRegistry with programmatic role support."""
    
    @pytest.fixture
    def temp_roles_dir(self):
        """Create temporary roles directory for testing."""
        temp_dir = tempfile.mkdtemp()
        roles_path = Path(temp_dir) / "roles"
        roles_path.mkdir()
        
        # Create a sample YAML role
        sample_role_dir = roles_path / "sample_llm_role"
        sample_role_dir.mkdir()
        
        definition_content = """
role:
  name: "sample_llm_role"
  execution_type: "llm"
  description: "Sample LLM-based role for testing"
  version: "1.0.0"
  
prompts:
  system: "You are a sample LLM role for testing."
  
tools:
  automatic: true
  shared: ["web_search"]
"""
        
        with open(sample_role_dir / "definition.yaml", "w") as f:
            f.write(definition_content)
        
        # Create shared_tools directory
        shared_tools_dir = roles_path / "shared_tools"
        shared_tools_dir.mkdir()
        (shared_tools_dir / "__init__.py").touch()
        
        yield roles_path
        shutil.rmtree(temp_dir)
    
    def test_enhanced_role_registry_initialization(self, temp_roles_dir):
        """Test enhanced RoleRegistry initialization with both role types."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Should have initialized both registries
        assert hasattr(registry, 'llm_roles')
        assert hasattr(registry, 'programmatic_roles')
        assert hasattr(registry, 'role_types')
        
        # Should have loaded the sample LLM role
        assert len(registry.llm_roles) >= 1
        assert "sample_llm_role" in registry.llm_roles
        assert registry.get_role_type("sample_llm_role") == "llm"
    
    def test_register_programmatic_role(self, temp_roles_dir):
        """Test registering programmatic roles."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Create and register a programmatic role
        mock_role = MockProgrammaticRole("test_programmatic", "Test programmatic role")
        registry.register_programmatic_role(mock_role)
        
        # Verify registration
        assert "test_programmatic" in registry.programmatic_roles
        assert registry.programmatic_roles["test_programmatic"] == mock_role
        assert registry.get_role_type("test_programmatic") == "programmatic"
    
    def test_register_llm_role(self, temp_roles_dir):
        """Test registering LLM-based roles."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Create a mock role definition
        mock_definition = RoleDefinition(
            name="test_llm_role",
            config={"role": {"name": "test_llm_role", "description": "Test LLM role"}},
            custom_tools=[],
            shared_tools={}
        )
        
        registry.register_llm_role("test_llm_role", mock_definition)
        
        # Verify registration
        assert "test_llm_role" in registry.llm_roles
        assert registry.llm_roles["test_llm_role"] == mock_definition
        assert registry.get_role_type("test_llm_role") == "llm"
    
    def test_get_role_type(self, temp_roles_dir):
        """Test role type detection."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register both types
        mock_prog_role = MockProgrammaticRole("prog_role", "Programmatic role")
        registry.register_programmatic_role(mock_prog_role)
        
        mock_llm_definition = RoleDefinition("llm_role", {}, [], {})
        registry.register_llm_role("llm_role", mock_llm_definition)
        
        # Test type detection
        assert registry.get_role_type("prog_role") == "programmatic"
        assert registry.get_role_type("llm_role") == "llm"
        assert registry.get_role_type("nonexistent_role") == "llm"  # Default
    
    def test_get_all_roles(self, temp_roles_dir):
        """Test getting all roles with their types."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register both types
        mock_prog_role = MockProgrammaticRole("prog_role", "Programmatic role")
        registry.register_programmatic_role(mock_prog_role)
        
        mock_llm_definition = RoleDefinition("llm_role", {}, [], {})
        registry.register_llm_role("llm_role", mock_llm_definition)
        
        all_roles = registry.get_all_roles()
        
        # Should include both types
        assert "prog_role" in all_roles
        assert "llm_role" in all_roles
        assert all_roles["prog_role"] == "programmatic"
        assert all_roles["llm_role"] == "llm"
        
        # Should also include the sample role from fixture
        assert "sample_llm_role" in all_roles
        assert all_roles["sample_llm_role"] == "llm"
    
    def test_get_programmatic_role(self, temp_roles_dir):
        """Test retrieving programmatic roles."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register a programmatic role
        mock_role = MockProgrammaticRole("test_prog", "Test programmatic role")
        registry.register_programmatic_role(mock_role)
        
        # Test retrieval
        retrieved_role = registry.get_programmatic_role("test_prog")
        assert retrieved_role == mock_role
        
        # Test non-existent role
        assert registry.get_programmatic_role("nonexistent") is None
    
    def test_get_llm_role(self, temp_roles_dir):
        """Test retrieving LLM roles (backward compatibility)."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Should be able to get existing LLM role
        llm_role = registry.get_role("sample_llm_role")
        assert llm_role is not None
        assert llm_role.name == "sample_llm_role"
        
        # Test with programmatic role (should return None)
        mock_prog_role = MockProgrammaticRole("prog_role", "Programmatic role")
        registry.register_programmatic_role(mock_prog_role)
        
        assert registry.get_role("prog_role") is None  # get_role is for LLM roles only
    
    def test_is_programmatic_role(self, temp_roles_dir):
        """Test checking if a role is programmatic."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register both types
        mock_prog_role = MockProgrammaticRole("prog_role", "Programmatic role")
        registry.register_programmatic_role(mock_prog_role)
        
        # Test detection
        assert registry.is_programmatic_role("prog_role") is True
        assert registry.is_programmatic_role("sample_llm_role") is False
        assert registry.is_programmatic_role("nonexistent") is False
    
    def test_list_programmatic_roles(self, temp_roles_dir):
        """Test listing all programmatic roles."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register multiple programmatic roles
        role1 = MockProgrammaticRole("prog_role_1", "First programmatic role")
        role2 = MockProgrammaticRole("prog_role_2", "Second programmatic role")
        
        registry.register_programmatic_role(role1)
        registry.register_programmatic_role(role2)
        
        # Test listing
        prog_roles = registry.list_programmatic_roles()
        assert len(prog_roles) == 2
        assert "prog_role_1" in prog_roles
        assert "prog_role_2" in prog_roles
        assert prog_roles["prog_role_1"] == role1
        assert prog_roles["prog_role_2"] == role2
    
    def test_get_role_metrics(self, temp_roles_dir):
        """Test getting metrics for programmatic roles."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register a programmatic role with some execution history
        mock_role = MockProgrammaticRole("metrics_role", "Role with metrics")
        mock_role.execution_count = 5
        mock_role.total_execution_time = 2.5
        
        registry.register_programmatic_role(mock_role)
        
        # Test metrics retrieval
        metrics = registry.get_role_metrics("metrics_role")
        assert metrics is not None
        assert metrics["name"] == "metrics_role"
        assert metrics["execution_count"] == 5
        assert metrics["total_execution_time"] == 2.5
        assert metrics["average_execution_time"] == 0.5
        
        # Test non-existent role
        assert registry.get_role_metrics("nonexistent") is None
    
    def test_enhanced_statistics(self, temp_roles_dir):
        """Test enhanced statistics including programmatic roles."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register programmatic roles
        role1 = MockProgrammaticRole("prog_1", "First programmatic role")
        role2 = MockProgrammaticRole("prog_2", "Second programmatic role")
        registry.register_programmatic_role(role1)
        registry.register_programmatic_role(role2)
        
        # Get enhanced statistics
        stats = registry.get_enhanced_statistics()
        
        # Should include programmatic role stats
        assert "total_programmatic_roles" in stats
        assert "total_llm_roles" in stats
        assert "role_type_distribution" in stats
        
        assert stats["total_programmatic_roles"] == 2
        assert stats["total_llm_roles"] >= 1  # At least the sample role
        
        distribution = stats["role_type_distribution"]
        assert "programmatic" in distribution
        assert "llm" in distribution
        assert distribution["programmatic"] == 2
    
    def test_backward_compatibility(self, temp_roles_dir):
        """Test that existing functionality remains unchanged."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Existing methods should still work
        assert hasattr(registry, 'get_role')
        assert hasattr(registry, 'get_role_summaries')
        assert hasattr(registry, 'list_roles')
        assert hasattr(registry, 'get_shared_tool')
        assert hasattr(registry, 'get_statistics')
        
        # Should still load YAML roles
        sample_role = registry.get_role("sample_llm_role")
        assert sample_role is not None
        
        # Existing statistics should work
        stats = registry.get_statistics()
        assert "total_roles" in stats
        assert stats["total_roles"] >= 1
    
    def test_role_validation_with_execution_type(self, temp_roles_dir):
        """Test role validation includes execution type checking."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Test validation of existing role
        validation = registry.validate_role("sample_llm_role")
        assert validation["valid"] is True
        
        # Register programmatic role and test validation
        mock_role = MockProgrammaticRole("prog_role", "Programmatic role")
        registry.register_programmatic_role(mock_role)
        
        # Programmatic roles should have different validation
        prog_validation = registry.validate_programmatic_role("prog_role")
        assert prog_validation["valid"] is True
        assert prog_validation["role_type"] == "programmatic"
    
    def test_error_handling_for_invalid_programmatic_roles(self, temp_roles_dir):
        """Test error handling when registering invalid programmatic roles."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Test registering None
        with pytest.raises((TypeError, AttributeError)):
            registry.register_programmatic_role(None)
        
        # Test registering non-ProgrammaticRole object
        with pytest.raises((TypeError, AttributeError)):
            registry.register_programmatic_role("not_a_role")
    
    def test_role_discovery_mixed_types(self, temp_roles_dir):
        """Test role discovery works with mixed role types."""
        registry = RoleRegistry(str(temp_roles_dir))
        
        # Register programmatic role
        mock_role = MockProgrammaticRole("prog_role", "Programmatic role")
        registry.register_programmatic_role(mock_role)
        
        # Get all roles should include both types
        all_roles = registry.get_all_roles()
        assert len(all_roles) >= 2  # At least sample_llm_role + prog_role
        
        # Should have both types represented
        role_types = set(all_roles.values())
        assert "llm" in role_types
        assert "programmatic" in role_types