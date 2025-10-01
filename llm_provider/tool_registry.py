from typing import Dict, List, Optional, Callable, Any
import inspect
from functools import wraps


def tool(name: Optional[str] = None, description: str = "", role: Optional[str] = None):
    """
    Decorator to register a function as a tool for the Universal Agent.
    
    Args:
        name: Optional name for the tool (uses function name if not provided)
        description: Description of what the tool does
        role: Optional role this tool is associated with
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        
        # Add tool metadata to the function
        func._tool_name = tool_name
        func._tool_description = description
        func._tool_role = role
        func._is_tool = True
        
        # Register with global registry
        ToolRegistry.get_global_registry().add_tool(tool_name, func, description, role)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class ToolRegistry:
    """
    Registry for managing tools available to the Universal Agent.
    """
    
    _global_registry = None
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.role_tools: Dict[str, List[str]] = {}
    
    @classmethod
    def get_global_registry(cls) -> 'ToolRegistry':
        """Get the global tool registry instance."""
        if cls._global_registry is None:
            cls._global_registry = cls()
        return cls._global_registry
    
    def add_tool(self, name: str, func: Callable, description: str = "", role: Optional[str] = None):
        """
        Add a tool to the registry.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            role: Optional role association
        """
        # Get function signature for validation
        sig = inspect.signature(func)
        
        tool_info = {
            'name': name,
            'function': func,
            'description': description,
            'role': role,
            'signature': sig,
            'parameters': list(sig.parameters.keys()),
            'return_annotation': sig.return_annotation
        }
        
        self.tools[name] = tool_info
        
        # Associate with role if specified
        if role:
            if role not in self.role_tools:
                self.role_tools[role] = []
            if name not in self.role_tools[role]:
                self.role_tools[role].append(name)
    
    def remove_tool(self, name: str):
        """
        Remove a tool from the registry.
        
        Args:
            name: Tool name to remove
        """
        if name in self.tools:
            tool_info = self.tools[name]
            role = tool_info.get('role')
            
            # Remove from tools
            del self.tools[name]
            
            # Remove from role association
            if role and role in self.role_tools:
                if name in self.role_tools[role]:
                    self.role_tools[role].remove(name)
                if not self.role_tools[role]:
                    del self.role_tools[role]
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool information dict or None if not found
        """
        return self.tools.get(name)
    
    def get_tools(self, tool_names: List[str]) -> List[Callable]:
        """
        Get tool functions by names.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            List of tool functions
        """
        tools = []
        for name in tool_names:
            if name in self.tools:
                tools.append(self.tools[name]['function'])
        return tools
    
    def get_tools_for_role(self, role: str) -> List[str]:
        """
        Get tool names associated with a role.
        
        Args:
            role: Role name
            
        Returns:
            List of tool names for the role
        """
        return self.role_tools.get(role, []).copy()
    
    def list_tools(self) -> List[str]:
        """
        Get list of all tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def list_roles(self) -> List[str]:
        """
        Get list of all roles that have associated tools.
        
        Returns:
            List of role names
        """
        return list(self.role_tools.keys())
    
    def search_tools(self, query: str) -> List[str]:
        """
        Search for tools by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching tool names
        """
        query_lower = query.lower()
        matches = []
        
        for name, tool_info in self.tools.items():
            if (query_lower in name.lower() or 
                query_lower in tool_info.get('description', '').lower()):
                matches.append(name)
        
        return matches
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Detailed tool information
        """
        if name not in self.tools:
            return None
        
        tool_info = self.tools[name].copy()
        
        # Convert signature to string for serialization
        if 'signature' in tool_info:
            tool_info['signature_str'] = str(tool_info['signature'])
            del tool_info['signature']  # Remove non-serializable signature
        
        return tool_info
    
    def validate_tool(self, name: str) -> bool:
        """
        Validate that a tool is properly configured.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool is valid
        """
        if name not in self.tools:
            return False
        
        tool_info = self.tools[name]
        
        # Check that function is callable
        if not callable(tool_info['function']):
            return False
        
        # Check that function has proper signature
        try:
            sig = tool_info['signature']
            # Tool should be callable with some parameters
            return True
        except Exception:
            return False
    
    def clear(self):
        """Clear all tools from the registry."""
        self.tools.clear()
        self.role_tools.clear()
    
    def export_tools(self) -> Dict[str, Any]:
        """
        Export tool registry for serialization.
        
        Returns:
            Serializable representation of tools
        """
        exported = {
            'tools': {},
            'role_tools': self.role_tools.copy()
        }
        
        for name, tool_info in self.tools.items():
            exported_tool = {
                'name': tool_info['name'],
                'description': tool_info['description'],
                'role': tool_info['role'],
                'parameters': tool_info['parameters'],
                'signature_str': str(tool_info['signature'])
            }
            exported['tools'][name] = exported_tool
        
        return exported
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Statistics about the tool registry
        """
        return {
            'total_tools': len(self.tools),
            'total_roles': len(self.role_tools),
            'tools_per_role': {role: len(tools) for role, tools in self.role_tools.items()},
            'tools_without_role': len([t for t in self.tools.values() if not t.get('role')])
        }


# Example tools for demonstration
@tool(name="create_task_plan", description="Create a comprehensive task plan", role="planning")
def create_task_plan(instruction: str, available_agents: List[str]) -> Dict:
    """Create a task plan from user instruction - converted from PlanningAgent."""
    return {
        "plan": f"Plan for: {instruction}",
        "steps": ["Step 1", "Step 2", "Step 3"],
        "agents": available_agents,
        "estimated_time": "2 hours"
    }


@tool(name="web_search", description="Search the web for information", role="search")
def web_search(query: str, num_results: int = 5) -> Dict:
    """Search the web for information - converted from SearchAgent."""
    return {
        "query": query,
        "results": [f"Result {i+1} for {query}" for i in range(num_results)],
        "total_results": num_results
    }


@tool(name="summarize_text", description="Summarize long text content", role="summarizer")
def summarize_text(text: str, max_length: int = 200) -> str:
    """Summarize text content - converted from SummarizerAgent."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


@tool(name="get_weather", description="Get weather information for a location", role="weather")
def get_weather(location: str) -> Dict:
    """Get weather information - converted from WeatherAgent."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "65%",
        "forecast": "Clear skies expected"
    }


@tool(name="send_slack_message", description="Send a message to Slack", role="slack")
def send_slack_message(channel: str, message: str) -> Dict:
    """Send Slack message - converted from SlackAgent."""
    return {
        "channel": channel,
        "message": message,
        "status": "sent",
        "timestamp": "2024-01-01T12:00:00Z"
    }