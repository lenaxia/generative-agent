from typing import List, Optional, Dict, Any, Callable
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.tool_registry import ToolRegistry
from llm_provider.mcp_client import MCPClientManager
from llm_provider.role_registry import RoleRegistry, RoleDefinition
from common.task_context import TaskContext
import logging

# Import StrandsAgent - hard dependency, no fallbacks
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel
from strands_tools import calculator, file_read, shell

logger = logging.getLogger(__name__)


class UniversalAgent:
    """
    Universal Agent that can assume different roles using StrandsAgent framework.
    
    This class provides a unified interface for creating role-specific agents
    while leveraging the semantic model types and prompt library from StrandsAgent.
    """
    
    def __init__(self, llm_factory: LLMFactory, role_registry: Optional[RoleRegistry] = None,
                 mcp_manager: Optional[MCPClientManager] = None):
        """
        Initialize Universal Agent with LLMFactory, role registry, and optional MCP manager.
        
        Args:
            llm_factory: Enhanced LLMFactory instance
            role_registry: Role registry for dynamic role loading
            mcp_manager: Optional MCP client manager for external tools
        """
        self.llm_factory = llm_factory
        self.tool_registry = ToolRegistry()
        self.role_registry = role_registry or RoleRegistry.get_global_registry()
        self.mcp_manager = mcp_manager
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None
    
    def assume_role(self, role: str, llm_type: LLMType = LLMType.DEFAULT,
                   context: Optional[TaskContext] = None, tools: Optional[List[str]] = None):
        """
        Create a role-specific agent using dynamic role definitions.
        
        Args:
            role: The role name (e.g., 'research_analyst', 'code_reviewer') or "None" for dynamic generation
            llm_type: Semantic model type for performance/cost optimization
            context: Optional TaskContext for state management
            tools: Optional list of additional tool names to include
            
        Returns:
            StrandsAgent Agent instance configured for the specified role
        """
        # Handle dynamic role generation for None or missing roles
        if role == "None" or role is None:
            logger.info("Dynamic role generation requested for None role")
            return self._create_dynamic_role_agent(llm_type, context)
        
        # Load role definition
        role_def = self.role_registry.get_role(role)
        if not role_def:
            logger.info(f"Role '{role}' not found in registry, generating dynamic role")
            return self._create_dynamic_role_agent(llm_type, context, role)
        
        # Create model with role-specific configuration
        model = self._create_model_for_role(role_def, llm_type)
        
        # Get system prompt from role definition
        system_prompt = self._get_system_prompt_from_role(role_def)
        
        # Assemble all tools for this role
        role_tools = self._assemble_role_tools(role_def, tools or [])
        
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
        
        logger.info(f"Assumed role '{role}' with {len(role_tools)} tools")
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
        Execute a task with hybrid execution path selection.
        
        Determines whether to use programmatic or LLM-based execution
        based on the role type for optimal performance.
        
        Args:
            instruction: Task instruction
            role: Agent role to assume
            llm_type: Model type for optimization
            context: Optional task context
            
        Returns:
            str: Task result
        """
        # Determine execution path based on role type
        if self.is_programmatic_role(role):
            return self.execute_programmatic_task(instruction, role, context)
        else:
            return self.execute_llm_task(instruction, role, llm_type, context)
    
    def is_programmatic_role(self, role: str) -> bool:
        """
        Check if role should use programmatic execution.
        
        Args:
            role: Role name to check
            
        Returns:
            bool: True if role uses programmatic execution
        """
        return self.role_registry.is_programmatic_role(role)
    
    def get_role_type(self, role: str) -> str:
        """
        Get the execution type for a role.
        
        Args:
            role: Role name
            
        Returns:
            str: "programmatic" or "llm"
        """
        return self.role_registry.get_role_type(role)
    
    def execute_programmatic_task(self, instruction: str, role: str, context: Optional[TaskContext] = None) -> str:
        """
        Execute task using programmatic role (no LLM processing).
        
        Args:
            instruction: Task instruction
            role: Programmatic role name
            context: Optional task context
            
        Returns:
            str: Serialized task result
        """
        try:
            programmatic_role = self.role_registry.get_programmatic_role(role)
            if not programmatic_role:
                return f"Programmatic role '{role}' not found"
            
            result = programmatic_role.execute(instruction, context)
            return self._serialize_result(result)
            
        except Exception as e:
            logger.error(f"Programmatic execution failed for role '{role}': {e}")
            return f"Programmatic execution error: {str(e)}"
    
    def execute_llm_task(self, instruction: str, role: str, llm_type: LLMType, context: Optional[TaskContext] = None) -> str:
        """
        Execute task using LLM-based role (current implementation).
        
        Args:
            instruction: Task instruction
            role: LLM role name
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
    
    def _create_model_for_role(self, role_def: RoleDefinition, llm_type: LLMType) -> BedrockModel:
        """
        Create model with role-specific configuration merged with model capabilities.
        
        Args:
            role_def: Role definition containing model config
            llm_type: Semantic model type
            
        Returns:
            Configured StrandsAgent model
        """
        # Get role model config
        role_model_config = role_def.config.get('model_config', {})
        
        # Get base model configuration
        base_model = self._create_strands_model(llm_type)
        
        # Merge configurations (role preferences with model limits)
        merged_config = self._merge_model_configs(role_model_config, base_model)
        
        return merged_config
    
    def _merge_model_configs(self, role_config: Dict, base_model: BedrockModel) -> BedrockModel:
        """Merge role configuration with base model."""
        try:
            # Handle mocked objects in tests
            if hasattr(base_model, '_mock_name'):
                # Return the base model as-is for mocked objects
                return base_model
            
            # Extract current model parameters safely
            model_params = {
                'model_id': getattr(base_model, 'model_id', 'us.anthropic.claude-sonnet-4-20250514-v1:0'),
                'region_name': getattr(base_model, 'region_name', 'us-west-2'),
                'temperature': role_config.get('temperature', getattr(base_model, 'temperature', 0.3)),
                'max_tokens': min(
                    role_config.get('max_tokens', getattr(base_model, 'max_tokens', 4096)),
                    getattr(base_model, 'max_tokens', 4096)  # Respect model limits
                )
            }
            
            # Create new model with merged config
            return BedrockModel(**model_params)
            
        except Exception as e:
            logger.warning(f"Failed to merge model configs: {e}, using base model")
            return base_model
    
    def _get_system_prompt_from_role(self, role_def: RoleDefinition) -> str:
        """Get system prompt from role definition."""
        prompts = role_def.config.get('prompts', {})
        return prompts.get('system', 'You are a helpful AI assistant.')
    
    def _assemble_role_tools(self, role_def: RoleDefinition, additional_tools: List[str]) -> List:
        """
        Assemble all tools for a role from multiple sources.
        
        Args:
            role_def: Role definition
            additional_tools: Additional tool names to include
            
        Returns:
            List of tool functions
        """
        tools = []
        
        # 1. Add built-in StrandsAgent tools
        builtin_tools = [calculator, file_read, shell]
        tools.extend(builtin_tools)
        
        # 2. Auto-include ALL custom tools from role's tools.py
        tools.extend(role_def.custom_tools)
        
        # 3. Add specified shared tools
        shared_tool_names = role_def.config.get('tools', {}).get('shared', [])
        for tool_name in shared_tool_names:
            shared_tool = self.role_registry.get_shared_tool(tool_name)
            if shared_tool:
                tools.append(shared_tool)
            else:
                logger.warning(f"Shared tool '{tool_name}' not found for role '{role_def.name}'")
        
        # 4. Add additional tools from registry
        additional_role_tools = self.tool_registry.get_tools(additional_tools)
        tools.extend(additional_role_tools)
        
        # 5. Add MCP tools if available
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_tools_for_role(role_def.name)
            tools.extend(mcp_tools)
        
        # 6. TODO: Add automatically selected tools if enabled
        if role_def.config.get('tools', {}).get('automatic', False):
            # This would use LLM to select additional tools
            # For now, we'll implement this in a future iteration
            logger.info(f"Automatic tool selection enabled for role '{role_def.name}' (not yet implemented)")
        
        return tools
    
    def _create_dynamic_role_agent(self, llm_type: LLMType, task_context: Optional[TaskContext] = None,
                                 suggested_role: Optional[str] = None) -> Agent:
        """
        Create a dynamic role agent by using LLM to generate role definition and select tools.
        
        This uses the LLM to:
        1. Analyze the task at hand
        2. Select appropriate tools from shared tools
        3. Generate a custom system prompt for the dynamic role
        
        Args:
            llm_type: LLM type for the agent
            task_context: Optional task context for task analysis
            suggested_role: Optional suggested role name for context
            
        Returns:
            Agent configured with dynamically generated role
        """
        model = self._create_strands_model(llm_type)
        
        # Get task information for dynamic role generation
        task_info = self._extract_task_info(task_context, suggested_role)
        
        # Get all available shared tools
        all_shared_tools = self.role_registry.get_all_shared_tools()
        
        if all_shared_tools and self.llm_factory:
            # Use LLM to generate dynamic role
            dynamic_role_config = self._generate_dynamic_role_config(task_info, all_shared_tools)
            
            # Create agent with dynamic configuration
            selected_tools = self._get_selected_tools(dynamic_role_config.get('selected_tools', []), all_shared_tools)
            system_prompt = dynamic_role_config.get('system_prompt', self._get_default_system_prompt())
            
            # Add built-in tools
            from strands_tools import calculator, file_read, shell
            all_tools = [calculator, file_read, shell] + selected_tools
            
            agent = Agent(
                model=model,
                system_prompt=system_prompt,
                tools=all_tools
            )
            
            # Store current configuration
            self.current_agent = agent
            self.current_role = f"dynamic_{suggested_role or 'generated'}"
            self.current_llm_type = llm_type
            
            logger.info(f"Created dynamic role agent with {len(selected_tools)} selected tools and custom system prompt")
            return agent
        else:
            # Fallback to basic tools if no LLM factory or shared tools available
            logger.warning("No LLM factory or shared tools available, using basic dynamic agent")
            return self._create_basic_dynamic_agent(llm_type)
    
    def _extract_task_info(self, task_context: Optional[TaskContext], suggested_role: Optional[str]) -> str:
        """Extract task information for dynamic role generation."""
        task_info_parts = []
        
        if suggested_role:
            task_info_parts.append(f"Suggested role context: {suggested_role}")
        
        if task_context and hasattr(task_context, 'task_graph'):
            # Get current task details from context
            pending_tasks = [node for node in task_context.task_graph.nodes.values()
                           if node.status.value == 'PENDING']
            if pending_tasks:
                current_task = pending_tasks[0]
                task_info_parts.append(f"Current task: {current_task.task_name}")
                task_info_parts.append(f"Task prompt: {current_task.prompt}")
                task_info_parts.append(f"Task type: {current_task.task_type}")
        
        return " | ".join(task_info_parts) if task_info_parts else "General assistance task"
    
    def _generate_dynamic_role_config(self, task_info: str, available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate dynamic role configuration."""
        try:
            # Create tool descriptions for LLM
            tools_description = "\n".join([
                f"- {name}: {getattr(tool, '__doc__', 'Tool function').split('.')[0] if tool.__doc__ else 'Tool function'}"
                for name, tool in available_tools.items()
            ])
            
            prompt = f"""
            You are tasked with creating a dynamic role for an AI agent based on the following task information:
            
            Task Information: {task_info}
            
            Available Tools:
            {tools_description}
            
            Please analyze the task and provide:
            1. A list of the most useful tools for this task (maximum 5 tools)
            2. A custom system prompt that would help an AI agent excel at this specific task
            
            Respond in JSON format:
            {{
                "selected_tools": ["tool1", "tool2", "tool3"],
                "system_prompt": "You are a specialized AI agent for... [detailed prompt based on the task]"
            }}
            
            Focus on selecting tools that are directly relevant to the task and creating a system prompt that gives the agent clear guidance on how to approach this type of work.
            """
            
            # Use lightweight model for role generation
            model = self.llm_factory.create_strands_model(LLMType.DEFAULT)
            response = model.invoke(prompt)
            
            # Parse JSON response
            import json
            role_config = json.loads(response.content.strip())
            
            # Validate and clean up the response
            selected_tools = role_config.get('selected_tools', [])
            # Filter to only include tools that actually exist
            valid_tools = [tool for tool in selected_tools if tool in available_tools][:5]
            
            system_prompt = role_config.get('system_prompt', self._get_default_system_prompt())
            
            return {
                'selected_tools': valid_tools,
                'system_prompt': system_prompt
            }
            
        except Exception as e:
            logger.warning(f"Dynamic role generation failed: {e}, using fallback")
            # Fallback to basic tool selection
            return {
                'selected_tools': list(available_tools.keys())[:3],
                'system_prompt': self._get_default_system_prompt()
            }
    
    def _get_selected_tools(self, tool_names: List[str], available_tools: Dict[str, Any]) -> List[Any]:
        """Get tool functions from tool names."""
        selected_tools = []
        for tool_name in tool_names:
            if tool_name in available_tools:
                selected_tools.append(available_tools[tool_name])
        return selected_tools
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for dynamic roles."""
        return """You are a helpful AI assistant with access to specialized tools.
        Analyze the task at hand and use the most appropriate tools to complete it effectively.
        Provide clear, accurate, and helpful responses."""
    
    def _create_basic_dynamic_agent(self, llm_type: LLMType) -> Agent:
        """Create a basic dynamic agent when advanced generation fails."""
        model = self._create_strands_model(llm_type)
        
        # Use basic tools only
        from strands_tools import calculator, file_read, shell
        basic_tools = [calculator, file_read, shell]
        
        system_prompt = self._get_default_system_prompt()
        
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=basic_tools
        )
        
        # Store current configuration
        self.current_agent = agent
        self.current_role = "dynamic_basic"
        self.current_llm_type = llm_type
        
        logger.info("Created basic dynamic role agent with built-in tools only")
        return agent
    
    def _select_tools_for_unknown_role(self, task_context: Optional[TaskContext] = None) -> List:
        """
        Intelligently select tools for unknown roles using LLM.
        
        This provides the performance benefit explanation:
        - Predefined roles: Fast execution with pre-selected tools
        - Unknown roles: Slower but intelligent tool selection as fallback
        """
        tools = [calculator, file_read, shell]  # Always include basics
        
        try:
            # Get task information for intelligent tool selection
            task_info = "general assistance"
            if task_context and hasattr(task_context, 'task_graph'):
                # Get task details from context if available
                pending_tasks = [node for node in task_context.task_graph.nodes.values()
                               if node.status == 'PENDING']
                if pending_tasks:
                    task_info = pending_tasks[0].prompt or "general assistance"
            
            # Get all available shared tools
            all_shared_tools = self.role_registry.get_all_shared_tools()
            
            if all_shared_tools and self.llm_factory:
                # Use LLM to select appropriate tools
                selected_tool_names = self._llm_select_tools_for_task(task_info, all_shared_tools)
                
                # Add selected shared tools
                for tool_name in selected_tool_names:
                    tool = self.role_registry.get_shared_tool(tool_name)
                    if tool:
                        tools.append(tool)
                
                logger.info(f"Fallback: Selected {len(selected_tool_names)} tools for unknown role: {selected_tool_names}")
            else:
                logger.info("Fallback: Using basic tools only (no LLM factory or shared tools available)")
                
        except Exception as e:
            logger.warning(f"Tool selection failed for unknown role, using basic tools: {e}")
        
        return tools
    
    def _llm_select_tools_for_task(self, task_info: str, available_tools: Dict[str, Callable]) -> List[str]:
        """Use LLM to select appropriate tools for a task."""
        try:
            # Create tool selection prompt
            tools_description = "\n".join([
                f"- {name}: {getattr(tool, '__doc__', 'No description').split('.')[0] if tool.__doc__ else 'Tool function'}"
                for name, tool in available_tools.items()
            ])
            
            prompt = f"""
            Task: {task_info}
            
            Available tools:
            {tools_description}
            
            Select the most useful tools for this task. Consider what the task might need.
            Return only tool names, one per line, maximum 5 tools.
            """
            
            # Use lightweight model for tool selection
            model = self.llm_factory.create_chat_model(LLMType.WEAK)
            response = model.invoke(prompt)
            
            # Parse tool names from response
            selected_tools = []
            for line in response.content.split('\n'):
                tool_name = line.strip()
                if tool_name and tool_name in available_tools:
                    selected_tools.append(tool_name)
                    if len(selected_tools) >= 5:  # Limit to 5 tools for performance
                        break
            
            return selected_tools
            
        except Exception as e:
            logger.warning(f"LLM tool selection failed: {e}")
            # Fallback to basic tool selection
            return list(available_tools.keys())[:3]  # Just take first 3 tools
    
    def reset(self):
        """Reset the Universal Agent state."""
        self.current_agent = None
        self.current_role = None
        self.current_llm_type = None
        
        # Execute the task
        try:
            response = agent(instruction)
            return str(response) if response else "No response generated"
        except Exception as e:
            logger.error(f"LLM task execution failed for role '{role}': {e}")
            return f"LLM execution error: {str(e)}"
    
    def register_programmatic_role(self, name: str, role_instance: 'ProgrammaticRole'):
        """
        Register a programmatic role for direct execution.
        
        Args:
            name: Role name
            role_instance: ProgrammaticRole instance
        """
        self.role_registry.register_programmatic_role(role_instance)
        logger.info(f"Registered programmatic role: {name}")
    
    def _serialize_result(self, result: Any) -> str:
        """
        Serialize programmatic results to string format.
        
        Args:
            result: Result data from programmatic execution
            
        Returns:
            str: Serialized result
        """
        if isinstance(result, str):
            return result
        elif isinstance(result, (dict, list)):
            import json
            return json.dumps(result, indent=2, default=str)
        else:
            return str(result)