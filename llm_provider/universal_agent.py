from typing import List, Optional, Dict, Any
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.tool_registry import ToolRegistry
from llm_provider.mcp_client import MCPClientManager
from common.task_context import TaskContext

# Import StrandsAgent - hard dependency, no fallbacks
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel
from strands_tools import calculator, file_read, shell


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
        # Create StrandsAgent model based on LLM type
        model = self._create_strands_model(llm_type)
        
        # Get role-specific system prompt
        system_prompt = self._get_role_prompt(role)
        
        # Get tools from registry and add built-in tools
        role_tools = self.tool_registry.get_tools(tools or [])
        
        # Add common Strands tools
        common_tools = [calculator, file_read, shell]
        role_tools.extend(common_tools)
        
        # Add role-specific @tool functions
        role_specific_tools = self._get_role_specific_tools(role)
        role_tools.extend(role_specific_tools)
        
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
    
    def _create_strands_model(self, llm_type: LLMType):
        """
        Create a StrandsAgent model based on LLM type.
        
        Args:
            llm_type: Semantic model type
            
        Returns:
            StrandsAgent model instance
        """
        # Map LLM types to model configurations from config
        model_mapping = {
            LLMType.WEAK: "us.anthropic.claude-sonnet-4-20250514-v1:0",
            LLMType.DEFAULT: "us.anthropic.claude-sonnet-4-20250514-v1:0", 
            LLMType.STRONG: "us.anthropic.claude-sonnet-4-20250514-v1:0"
        }
        
        model_id = model_mapping.get(llm_type, model_mapping[LLMType.DEFAULT])
        
        # Create Bedrock model with proper configuration
        return BedrockModel(
            model_id=model_id,
            region_name="us-west-2",
            temperature=0.3,
            max_tokens=4096
        )
    
    def _get_role_prompt(self, role: str) -> str:
        """
        Get role-specific system prompt.
        
        Args:
            role: Agent role
            
        Returns:
            str: System prompt for the role
        """
        role_prompts = {
            "planning": """You are a planning specialist agent. Your role is to:
1. Break down complex tasks into manageable steps
2. Create detailed task plans with dependencies
3. Identify required resources and constraints
4. Provide structured planning output
Focus on creating clear, actionable plans.""",
            
            "search": """You are a search specialist agent. Your role is to:
1. Perform web searches for information
2. Find relevant and accurate information
3. Summarize search results clearly
4. Provide source citations
Focus on finding the most relevant and up-to-date information.""",
            
            "weather": """You are a weather information specialist agent. Your role is to:
1. Retrieve current weather conditions
2. Provide weather forecasts
3. Explain weather patterns and phenomena
4. Give location-specific weather data
Focus on providing accurate, current weather information.""",
            
            "summarizer": """You are a text summarization specialist agent. Your role is to:
1. Create concise summaries of long texts
2. Extract key points and main ideas
3. Maintain important context and details
4. Provide structured summary output
Focus on creating clear, comprehensive summaries.""",
            
            "slack": """You are a Slack integration specialist agent. Your role is to:
1. Send messages to Slack channels
2. Format messages appropriately for Slack
3. Handle Slack-specific formatting and mentions
4. Manage Slack workspace interactions
Focus on effective Slack communication.""",
            
            "coding": """You are a coding specialist agent. Your role is to:
1. Write clean, efficient code
2. Debug and fix code issues
3. Explain code functionality
4. Follow best practices and patterns
Focus on producing high-quality, maintainable code.""",
            
            "analysis": """You are an analysis specialist agent. Your role is to:
1. Analyze data and information thoroughly
2. Identify patterns and insights
3. Provide detailed analytical reports
4. Make data-driven recommendations
Focus on comprehensive, accurate analysis."""
        }
        
        return role_prompts.get(role, 
            "You are a helpful AI assistant. Provide accurate, helpful responses to user queries.")
    
    def execute_task(self, instruction: str, role: str = "default", 
                    llm_type: LLMType = LLMType.DEFAULT,
                    context: Optional[TaskContext] = None) -> str:
        """
        Execute a task with the specified role and model type.
        
        Args:
            instruction: Task instruction
            role: Agent role to assume
            llm_type: Model type for optimization
            context: Optional task context
            
        Returns:
            str: Task result
        """
        # Assume the specified role
        agent = self.assume_role(role, llm_type, context)
        
        # Execute the task
        try:
            response = agent(instruction)
            return str(response) if response else "No response generated"
        except Exception as e:
            return f"Error executing task: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current Universal Agent status.
        
        Returns:
            Dict: Status information
        """
        return {
            "universal_agent_enabled": True,
            "has_llm_factory": self.llm_factory is not None,
            "has_universal_agent": True,
            "mcp_integration": {
                "mcp_available": self.mcp_manager is not None,
                "registered_servers": self.mcp_manager.get_registered_servers() if self.mcp_manager else [],
                "total_tools": len(self.mcp_manager.get_all_tools()) if self.mcp_manager else 0,
                "server_configs": self.mcp_manager.get_server_configs() if self.mcp_manager else {}
            },
            "current_role": self.current_role,
            "current_llm_type": self.current_llm_type.value if self.current_llm_type else None,
            "framework": "strands"
        }
    
    def get_available_roles(self) -> List[str]:
        """
        Get list of available agent roles.
        
        Returns:
            List[str]: Available roles
        """
        return ["planning", "search", "weather", "summarizer", "slack", "coding", "analysis", "default"]
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """
        Get MCP integration status for heartbeat monitoring.
        
        Returns:
            Dict: MCP status information
        """
        if self.mcp_manager:
            return {
                "mcp_available": True,
                "registered_servers": self.mcp_manager.get_registered_servers(),
                "total_tools": len(self.mcp_manager.get_all_tools()),
                "server_configs": self.mcp_manager.get_server_configs()
            }
        else:
            return {
                "mcp_available": False,
                "registered_servers": [],
                "total_tools": 0,
                "server_configs": {}
            }
    
    def _get_role_specific_tools(self, role: str) -> list:
        """
        Get role-specific @tool functions for the Universal Agent.
        
        Args:
            role: The agent role
            
        Returns:
            List of @tool functions for the role
        """
        role_tools = []
        
        try:
            if role == "weather":
                from llm_provider.weather_tools import get_weather
                # Convert function to @tool format if needed
                role_tools.append(get_weather)
                
            elif role == "search":
                from llm_provider.search_tools import search_web
                role_tools.append(search_web)
                
            elif role == "summarizer":
                from llm_provider.summarizer_tools import summarize_text
                role_tools.append(summarize_text)
                
            elif role == "slack":
                from llm_provider.slack_tools import send_slack_message
                role_tools.append(send_slack_message)
                
            elif role == "planning":
                from llm_provider.planning_tools import create_task_plan, validate_task_plan
                role_tools.extend([create_task_plan, validate_task_plan])
                
        except ImportError as e:
            # If tools don't exist or have import issues, log and continue
            logger.warning(f"Could not load tools for role '{role}': {e}")
        
        return role_tools
    
    def reset(self):
        """Reset the Universal Agent state."""
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None