"""
Role Registry Module

Manages dynamic loading and registration of roles from file-based definitions.
"""

from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import yaml
import importlib.util
import inspect
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoleDefinition:
    """Container for a complete role definition."""
    name: str
    config: Dict[str, Any]
    custom_tools: List[Callable]
    shared_tools: Dict[str, Callable]


class RoleRegistry:
    """
    Registry for managing dynamically loaded roles from file definitions.
    """
    
    _global_registry = None
    
    def __init__(self, roles_directory: str = "roles"):
        """
        Initialize the role registry.
        
        Args:
            roles_directory: Path to the roles directory
        """
        self.roles_directory = Path(roles_directory)
        self.roles: Dict[str, RoleDefinition] = {}
        self.shared_tools: Dict[str, Callable] = {}
        
        # Load shared tools first
        self._load_shared_tools()
        
        # Discover and load all roles
        self.refresh()
    
    @classmethod
    def get_global_registry(cls, roles_directory: str = "roles") -> 'RoleRegistry':
        """Get the global role registry instance."""
        if cls._global_registry is None:
            cls._global_registry = cls(roles_directory)
        return cls._global_registry
    
    def refresh(self):
        """Refresh role registry by discovering and loading all roles from filesystem."""
        logger.info("Refreshing role registry...")
        
        discovered_roles = self._discover_roles()
        logger.info(f"Discovered {len(discovered_roles)} roles: {discovered_roles}")
        
        for role_name in discovered_roles:
            try:
                role_def = self._load_role(role_name)
                self.roles[role_name] = role_def
                logger.info(f"Loaded role '{role_name}' with {len(role_def.custom_tools)} custom tools")
            except Exception as e:
                logger.error(f"Failed to load role '{role_name}': {e}")
        
        logger.info(f"Role registry refreshed with {len(self.roles)} roles")
    
    def _discover_roles(self) -> List[str]:
        """Discover all available role definitions."""
        roles = []
        
        if not self.roles_directory.exists():
            logger.warning(f"Roles directory not found: {self.roles_directory}")
            return roles
        
        for role_dir in self.roles_directory.iterdir():
            if role_dir.is_dir() and (role_dir / "definition.yaml").exists():
                # Skip shared_tools directory
                if role_dir.name != "shared_tools":
                    roles.append(role_dir.name)
        
        return roles
    
    def _load_role(self, role_name: str) -> RoleDefinition:
        """Load a single role definition from files."""
        role_path = self.roles_directory / role_name
        definition_file = role_path / "definition.yaml"
        
        # Load role configuration
        with open(definition_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load custom tools if tools.py exists
        custom_tools = []
        tools_file = role_path / "tools.py"
        if tools_file.exists():
            custom_tools = self._load_custom_tools(tools_file)
        
        return RoleDefinition(
            name=role_name,
            config=config,
            custom_tools=custom_tools,
            shared_tools=self.shared_tools
        )
    
    def _load_shared_tools(self):
        """Load shared tools from shared_tools directory."""
        shared_tools_dir = self.roles_directory / "shared_tools"
        
        if not shared_tools_dir.exists():
            logger.warning("Shared tools directory not found")
            return
        
        # Load tools from each shared tool file
        for tool_file in shared_tools_dir.glob("*.py"):
            if tool_file.name == "__init__.py":
                continue
            
            try:
                tools = self._load_tools_from_file(tool_file)
                for tool in tools:
                    if hasattr(tool, '_tool_name'):
                        self.shared_tools[tool._tool_name] = tool
                    else:
                        self.shared_tools[tool.__name__] = tool
                
                logger.info(f"Loaded {len(tools)} shared tools from {tool_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load shared tools from {tool_file}: {e}")
    
    def _load_custom_tools(self, tools_file: Path) -> List[Callable]:
        """Load all @tool functions from a role's tools.py file."""
        return self._load_tools_from_file(tools_file)
    
    def _load_tools_from_file(self, tools_file: Path) -> List[Callable]:
        """Load all @tool decorated functions from a Python file."""
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(f"tools_{tools_file.stem}", tools_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all @tool decorated functions
            tools = []
            for name, obj in inspect.getmembers(module):
                # Check for StrandsAgent @tool decorator with multiple detection methods
                if callable(obj) and (
                    hasattr(obj, '_is_tool') or  # Original check
                    hasattr(obj, '__wrapped__') or  # Decorator wrapper
                    hasattr(obj, '_tool_name') or  # StrandsAgent tool name
                    getattr(obj, '__name__', '').startswith('tool_') or  # Tool prefix
                    hasattr(obj, '_strands_tool') or  # StrandsAgent marker
                    str(type(obj)).find('tool') != -1  # Tool in type name
                ):
                    tools.append(obj)
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to load tools from {tools_file}: {e}")
            return []
    
    def get_role(self, role_name: str) -> Optional[RoleDefinition]:
        """
        Get role definition by name.
        
        Args:
            role_name: Name of the role
            
        Returns:
            RoleDefinition or None if not found
        """
        return self.roles.get(role_name)
    
    def get_role_summaries(self) -> List[Dict[str, str]]:
        """
        Get role summaries for planning LLM.
        
        Returns:
            List of role summaries with name, description, and when_to_use
        """
        summaries = []
        for name, role_def in self.roles.items():
            role_config = role_def.config.get('role', {})
            summaries.append({
                'name': name,
                'description': role_config.get('description', ''),
                'when_to_use': role_config.get('when_to_use', '')
            })
        return summaries
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """
        List all available roles with metadata.
        
        Returns:
            List of role information dicts
        """
        roles_info = []
        for name, role_def in self.roles.items():
            role_config = role_def.config.get('role', {})
            roles_info.append({
                "name": name,
                "description": role_config.get('description', ''),
                "version": role_config.get('version', '1.0.0'),
                "author": role_config.get('author', ''),
                "custom_tool_count": len(role_def.custom_tools),
                "has_automatic_tools": role_def.config.get('tools', {}).get('automatic', False),
                "shared_tools": role_def.config.get('tools', {}).get('shared', [])
            })
        return roles_info
    
    def get_shared_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a shared tool by name.
        
        Args:
            tool_name: Name of the shared tool
            
        Returns:
            Tool function or None if not found
        """
        return self.shared_tools.get(tool_name)
    
    def get_all_shared_tools(self) -> Dict[str, Callable]:
        """Get all shared tools."""
        return self.shared_tools.copy()
    
    def validate_role(self, role_name: str) -> Dict[str, Any]:
        """
        Validate a role definition.
        
        Args:
            role_name: Name of the role to validate
            
        Returns:
            Validation result with status and any errors
        """
        if role_name not in self.roles:
            return {
                "valid": False,
                "errors": [f"Role '{role_name}' not found"],
                "warnings": []
            }
        
        role_def = self.roles[role_name]
        errors = []
        warnings = []
        
        # Check required fields
        role_config = role_def.config.get('role', {})
        required_fields = ['name', 'description']
        for field in required_fields:
            if not role_config.get(field):
                errors.append(f"Missing required field: role.{field}")
        
        # Check prompts
        prompts = role_def.config.get('prompts', {})
        if not prompts.get('system'):
            warnings.append("No system prompt defined")
        
        # Check tools configuration
        tools_config = role_def.config.get('tools', {})
        if not tools_config.get('automatic') and not tools_config.get('shared') and not role_def.custom_tools:
            warnings.append("No tools configured (no automatic, shared, or custom tools)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "role_name": role_name
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Statistics about the role registry
        """
        total_custom_tools = sum(len(role_def.custom_tools) for role_def in self.roles.values())
        roles_with_automatic = sum(1 for role_def in self.roles.values() 
                                 if role_def.config.get('tools', {}).get('automatic', False))
        
        return {
            'total_roles': len(self.roles),
            'total_shared_tools': len(self.shared_tools),
            'total_custom_tools': total_custom_tools,
            'roles_with_automatic_tools': roles_with_automatic,
            'roles_with_custom_tools': sum(1 for role_def in self.roles.values() if role_def.custom_tools),
            'average_tools_per_role': total_custom_tools / len(self.roles) if self.roles else 0
        }