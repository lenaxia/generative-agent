from typing import List, Optional, Dict, Any
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.tool_registry import ToolRegistry
from llm_provider.mcp_client import MCPClientManager
from common.task_context import TaskContext

# Import StrandsAgent with fallback for testing
try:
    from strands import Agent
    from strands.models.bedrock import BedrockModel
    from strands.models.openai import OpenAIModel
    STRANDS_AVAILABLE = True
except ImportError:
    # Mock classes for testing environments
    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            
        def __call__(self, prompt):
            return f"Mock Agent response to: {prompt}"
    
    class BedrockModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    
    class OpenAIModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    
    STRANDS_AVAILABLE = False


class UniversalAgent:
    """
    Universal Agent that can assume different roles using StrandsAgent framework.
    
    This class provides a unified interface for creating role-specific agents
    while leveraging the semantic model types and prompt library from StrandsAgent.
    """
    
    def __init__(self, llm_factory: LLMFactory, mcp_manager: Optional[MCPClientManager] = None):
        """
        Initialize Universal Agent with an LLMFactory and optional MCP manager.
        
        Args:
            llm_factory: Enhanced LLMFactory instance
            mcp_manager: Optional MCP client manager for external tools
        """
        self.llm_factory = llm_factory
        self.tool_registry = ToolRegistry()
        self.mcp_manager = mcp_manager
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None
        self.strands_available = STRANDS_AVAILABLE
    
    def assume_role(self, role: str, llm_type: LLMType = LLMType.DEFAULT,
                   context: Optional[TaskContext] = None, tools: Optional[List[str]] = None):
        """
        Create a role-specific agent using StrandsAgent framework.
        
        Args:
            role: The agent role (e.g., 'planning', 'search', 'summarizer')
            llm_type: Semantic model type for performance/cost optimization
            context: Optional TaskContext for state management
            tools: Optional list of tool names to include
            
        Returns:
            StrandsAgent Agent instance configured for the specified role
        """
        if not self.strands_available:
            # Fallback for testing environments
            agent = Agent(role=role, llm_type=llm_type, tools=tools)
            self.current_agent = agent
            self.current_role = role
            self.current_llm_type = llm_type
            return agent
        
        # Create StrandsAgent model based on LLM type
        model = self._create_strands_model(llm_type)
        
        # Get role-specific system prompt
        system_prompt = self._get_role_prompt(role)
        
        # Get tools from registry
        role_tools = self.tool_registry.get_tools(tools or [])
        
        # Add MCP tools if available
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_tools_for_role(role)
            role_tools.extend(mcp_tools)
        
        # Create StrandsAgent Agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
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
    
    def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool with given parameters.
        
        Args:
            tool_name: Name of the MCP tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If MCP manager not available or tool not found
        """
        if not self.mcp_manager:
            raise ValueError("MCP manager not available")
        
        return self.mcp_manager.execute_tool(tool_name, parameters)
    
    def get_mcp_tools(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available MCP tools for a role.
        
        Args:
            role: Optional role to filter tools for (uses current role if not specified)
            
        Returns:
            List of available MCP tools
        """
        if not self.mcp_manager:
            return []
        
        target_role = role or self.current_role or "default"
        return self.mcp_manager.get_tools_for_role(target_role)
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """
        Get MCP integration status.
        
        Returns:
            Dict containing MCP status information
        """
        if not self.mcp_manager:
            return {"mcp_available": False, "servers": [], "tools": 0}
        
        return self.mcp_manager.get_server_status()
    
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
            "framework": self.llm_factory.get_framework(),
            "mcp_status": self.get_mcp_status()
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
    
    def _create_strands_model(self, llm_type: LLMType):
        """
        Create a StrandsAgent model based on LLM type.
        
        Args:
            llm_type: The semantic LLM type
            
        Returns:
            StrandsAgent model instance
        """
        if not self.strands_available:
            # Return mock model for testing
            return BedrockModel(model_id="mock-model")
        
        # Get configuration from factory
        config = self.llm_factory._get_config(llm_type)
        
        if hasattr(config, 'provider_type'):
            provider_type = config.provider_type
        elif hasattr(config, 'provider_name'):
            provider_type = config.provider_name
        else:
            provider_type = "bedrock"  # Default fallback
        
        # Extract model parameters
        model_params = {
            'model_id': getattr(config, 'model_id', 'us.amazon.nova-pro-v1:0'),
            'temperature': getattr(config, 'temperature', 0.3)
        }
        
        # Add additional parameters if available
        if hasattr(config, 'additional_params') and config.additional_params:
            model_params.update(config.additional_params)
        
        # Create appropriate StrandsAgent model
        if provider_type == "bedrock":
            return BedrockModel(**model_params)
        elif provider_type == "openai":
            return OpenAIModel(**model_params)
        else:
            # Default to Bedrock
            return BedrockModel(**model_params)
    
    def _get_role_prompt(self, role: str) -> str:
        """
        Get system prompt for a role.
        
        Args:
            role: The agent role
            
        Returns:
            str: System prompt for the role
        """
        try:
            return self.llm_factory.prompt_library.get_prompt(role)
        except:
            # Fallback prompts for different roles
            role_prompts = {
                "planning": "You are a planning assistant. Create detailed task plans with dependencies and agent assignments.",
                "search": "You are a search assistant. Help users find information efficiently.",
                "weather": "You are a weather assistant. Provide accurate weather information.",
                "summarizer": "You are a summarization assistant. Create concise, accurate summaries.",
                "slack": "You are a Slack assistant. Help with team communication and collaboration.",
                "default": "You are a helpful assistant."
            }
            return role_prompts.get(role, role_prompts["default"])