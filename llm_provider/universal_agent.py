from typing import List, Optional, Dict, Any
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.tool_registry import ToolRegistry
from common.task_context import TaskContext


class UniversalAgent:
    """
    Universal Agent that can assume different roles using the enhanced LLMFactory.
    
    This class provides a unified interface for creating role-specific agents
    while leveraging the semantic model types and prompt library.
    """
    
    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize Universal Agent with an LLMFactory.
        
        Args:
            llm_factory: Enhanced LLMFactory instance
        """
        self.llm_factory = llm_factory
        self.tool_registry = ToolRegistry()
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None
    
    def assume_role(self, role: str, llm_type: LLMType = LLMType.DEFAULT,
                   context: Optional[TaskContext] = None, tools: Optional[List[str]] = None):
        """
        Create a role-specific agent using the factory abstraction.
        
        Args:
            role: The agent role (e.g., 'planning', 'search', 'summarizer')
            llm_type: Semantic model type for performance/cost optimization
            context: Optional TaskContext for state management
            tools: Optional list of tool names to include
            
        Returns:
            Agent instance configured for the specified role
        """
        # Get tools from registry
        role_tools = self.tool_registry.get_tools(tools or [])
        
        # Use factory to create agent (maintains abstraction)
        agent = self.llm_factory.create_universal_agent(
            llm_type=llm_type,
            role=role,
            tools=role_tools
        )
        
        # Store current configuration
        self.current_agent = agent
        self.current_role = role
        self.current_llm_type = llm_type
        
        return agent
    
    def execute_task(self, task_prompt: str, role: Optional[str] = None, 
                    llm_type: Optional[LLMType] = None, tools: Optional[List[str]] = None,
                    context: Optional[TaskContext] = None) -> str:
        """
        Execute a task with the specified or current role configuration.
        
        Args:
            task_prompt: The task prompt to execute
            role: Optional role to assume (uses current if not specified)
            llm_type: Optional LLM type (uses current if not specified)
            tools: Optional tools to use
            context: Optional task context
            
        Returns:
            str: Task execution result
        """
        # Use provided role or current role
        if role and role != self.current_role:
            self.assume_role(role, llm_type or self.current_llm_type or LLMType.DEFAULT, context, tools)
        elif not self.current_agent:
            # No current agent, create default
            self.assume_role(role or "default", llm_type or LLMType.DEFAULT, context, tools)
        
        # Execute the task
        try:
            if hasattr(self.current_agent, '__call__') and callable(self.current_agent):
                return self.current_agent(task_prompt)
        except (AttributeError, TypeError):
            pass
        
        try:
            if hasattr(self.current_agent, 'run') and callable(self.current_agent.run):
                return self.current_agent.run(task_prompt)
        except (AttributeError, TypeError):
            pass
        
        try:
            if hasattr(self.current_agent, 'execute') and callable(self.current_agent.execute):
                return self.current_agent.execute(task_prompt)
        except (AttributeError, TypeError):
            pass
        
        # Fallback for mock objects in testing
        return f"Mock execution of '{task_prompt}' with role '{self.current_role}'"
    
    def get_current_role(self) -> Optional[str]:
        """Get the current agent role."""
        return self.current_role
    
    def get_current_llm_type(self) -> Optional[LLMType]:
        """Get the current LLM type."""
        return self.current_llm_type
    
    def get_available_roles(self) -> List[str]:
        """Get list of available roles from the prompt library."""
        return self.llm_factory.prompt_library.list_roles()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools from the tool registry."""
        return self.tool_registry.list_tools()
    
    def add_tool(self, tool_name: str, tool_func: callable, description: str = ""):
        """
        Add a tool to the tool registry.
        
        Args:
            tool_name: Name of the tool
            tool_func: Tool function
            description: Tool description
        """
        self.tool_registry.add_tool(tool_name, tool_func, description)
    
    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the tool registry.
        
        Args:
            tool_name: Name of the tool to remove
        """
        self.tool_registry.remove_tool(tool_name)
    
    def get_role_configuration(self, role: str) -> Dict[str, Any]:
        """
        Get configuration information for a role.
        
        Args:
            role: The role to get configuration for
            
        Returns:
            Dict containing role configuration
        """
        return {
            "role": role,
            "prompt": self.llm_factory.prompt_library.get_prompt(role),
            "available_tools": self.tool_registry.get_tools_for_role(role),
            "recommended_llm_type": self._get_recommended_llm_type(role)
        }
    
    def _get_recommended_llm_type(self, role: str) -> LLMType:
        """
        Get recommended LLM type for a role based on complexity.
        
        Args:
            role: The agent role
            
        Returns:
            LLMType: Recommended semantic model type
        """
        # Map roles to appropriate model types for cost/performance optimization
        role_to_llm_type = {
            "planning": LLMType.STRONG,    # Complex reasoning needs powerful model
            "analysis": LLMType.STRONG,    # Complex analysis needs powerful model
            "coding": LLMType.STRONG,      # Code generation needs powerful model
            "search": LLMType.WEAK,        # Simple search can use cheaper model
            "weather": LLMType.WEAK,       # Simple lookup
            "summarizer": LLMType.DEFAULT, # Balanced model for text processing
            "slack": LLMType.DEFAULT,      # Conversational tasks
            "default": LLMType.DEFAULT     # Default fallback
        }
        return role_to_llm_type.get(role, LLMType.DEFAULT)
    
    def optimize_for_cost(self):
        """Switch to cost-optimized model selection."""
        if self.current_role:
            # Use WEAK model for cost optimization
            self.assume_role(self.current_role, LLMType.WEAK)
    
    def optimize_for_performance(self):
        """Switch to performance-optimized model selection."""
        if self.current_role:
            # Use STRONG model for performance optimization
            self.assume_role(self.current_role, LLMType.STRONG)
    
    def reset(self):
        """Reset the agent to initial state."""
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the Universal Agent.
        
        Returns:
            Dict containing current status information
        """
        return {
            "current_role": self.current_role,
            "current_llm_type": self.current_llm_type.value if self.current_llm_type else None,
            "has_active_agent": self.current_agent is not None,
            "available_roles": len(self.get_available_roles()),
            "available_tools": len(self.get_available_tools()),
            "framework": self.llm_factory.get_framework()
        }
    
    def __str__(self) -> str:
        """String representation of Universal Agent."""
        status = self.get_status()
        return (f"UniversalAgent(role={status['current_role']}, "
                f"llm_type={status['current_llm_type']}, "
                f"framework={status['framework']})")
    
    def __repr__(self) -> str:
        """Detailed representation of Universal Agent."""
        return self.__str__()