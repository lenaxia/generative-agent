from enum import Enum
from typing import Optional, Dict, List, Union, Type, Any
import logging
import hashlib
from config.base_config import BaseConfig

# Import StrandsAgent dependencies with fallback for testing
try:
    from strands.models.bedrock import BedrockModel
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError as e:
    # Mock classes for testing when StrandsAgent is not available
    class BedrockModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    
    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        
        def __call__(self, instruction):
            return f"Mock agent response to: {instruction}"
    
    STRANDS_AVAILABLE = False

# Try to import OpenAI model separately since it might not be available
try:
    from strands.models.openai import OpenAIModel
    OPENAI_AVAILABLE = True
except ImportError:
    class OpenAIModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    OPENAI_AVAILABLE = False

from pydantic import BaseModel
from llm_provider.prompt_library import PromptLibrary

logger = logging.getLogger(__name__)

# Log import status after logger is defined
if STRANDS_AVAILABLE:
    logger.info("Successfully imported Strands components")
else:
    logger.warning("Strands import failed, using mock components")

class LLMType(Enum):
    DEFAULT = 'default'
    STRONG = 'strong'
    WEAK = 'weak'
    VISION = 'vision'
    MULTIMODAL = 'multimodal'
    IMAGE_GENERATION = 'image_generation'

class LLMFactory:
    def __init__(self, configs: Dict[LLMType, List[BaseConfig]], framework: str = "strands"):
        """
        Initialize LLMFactory with StrandsAgent support and performance caching.
        
        Args:
            configs: Configuration mapping for different LLM types
            framework: Framework to use (only "strands" supported)
        """
        self.configs = configs
        self.framework = "strands"  # Only StrandsAgent framework supported
        self.prompt_library = PromptLibrary()
        
        # Performance optimization: Model and agent caching
        self._model_cache: Dict[str, Any] = {}
        self._agent_cache: Dict[str, Any] = {}
        self._is_warmed = False
        
        # Enhanced caching infrastructure for agent pooling
        self._agent_pool: Dict[str, Any] = {}  # Provider-based Agent pool
        self._pool_stats = {'hits': 0, 'misses': 0, 'created': 0}
        
        logger.info("LLMFactory initialized with caching and agent pooling enabled")

    def add_config(self, llm_type: LLMType, config: BaseConfig):
        """Add a configuration for the specified LLM type."""
        if llm_type not in self.configs:
            self.configs[llm_type] = []
        self.configs[llm_type].append(config)

    def remove_config(self, llm_type: LLMType, name: str):
        """Remove a configuration by name for the specified LLM type."""
        if llm_type not in self.configs:
            return
        self.configs[llm_type] = [config for config in self.configs[llm_type] if config.name != name]

    def create_strands_model(self, llm_type: LLMType, name: Optional[str] = None):
        """
        Create a StrandsAgent model instance with caching for performance.
        
        Args:
            llm_type: The semantic LLM type (DEFAULT, STRONG, WEAK, etc.)
            name: Optional specific configuration name
            
        Returns:
            StrandsAgent model instance (cached if available)
            
        Raises:
            ValueError: If configuration not found or unsupported provider
        """
        # Create cache key based on llm_type and name
        cache_key = f"{llm_type.value}_{name or 'default'}"
        
        # Return cached model if available
        if cache_key in self._model_cache:
            logger.debug(f"Returning cached model for {cache_key}")
            return self._model_cache[cache_key]
        
        # Create new model
        logger.debug(f"Creating new model for {cache_key}")
        model = self._create_new_model(llm_type, name)
        
        # Cache the model
        self._model_cache[cache_key] = model
        logger.info(f"Cached new model for {cache_key}")
        
        return model
    
    def _create_new_model(self, llm_type: LLMType, name: Optional[str] = None):
        """
        Internal method to create a new model instance.
        
        Args:
            llm_type: The semantic LLM type
            name: Optional specific configuration name
            
        Returns:
            New StrandsAgent model instance
        """
        config = self._get_config(llm_type, name)
        
        if hasattr(config, 'provider_type'):
            provider_type = config.provider_type
        elif hasattr(config, 'provider_name'):
            provider_type = config.provider_name
        else:
            raise ValueError("Configuration missing provider type information")
        
        # Extract model parameters
        model_params = {
            'model_id': getattr(config, 'model_id', None),
            'temperature': getattr(config, 'temperature', 0.3)
        }
        
        # Add additional parameters if available
        if hasattr(config, 'additional_params') and config.additional_params:
            model_params.update(config.additional_params)
        
        # Create appropriate model based on provider type
        if provider_type == "bedrock":
            return BedrockModel(**model_params)
        elif provider_type == "openai":
            if OPENAI_AVAILABLE:
                return OpenAIModel(**model_params)
            else:
                raise ValueError("OpenAI model requested but not available in Strands installation")
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    def create_universal_agent(self, llm_type: LLMType, role: str, tools: Optional[List] = None, role_registry=None):
        """
        Create a Universal Agent using StrandsAgent framework with caching.
        
        Args:
            llm_type: The semantic LLM type for model selection
            role: The agent role (determines system prompt)
            tools: Optional list of tools for the agent
            role_registry: Optional role registry for getting role-specific prompts
            
        Returns:
            StrandsAgent Agent instance (cached if available)
        """
        # Create cache key including role and tools hash for proper isolation
        tools_hash = self._hash_tools(tools or [])
        cache_key = f"{llm_type.value}_{role}_{tools_hash}"
        
        # Return cached agent if available
        if cache_key in self._agent_cache:
            logger.debug(f"Returning cached agent for {cache_key}")
            return self._agent_cache[cache_key]
        
        # Create new agent
        logger.debug(f"Creating new agent for {cache_key}")
        
        # Create the model (will use model cache)
        model = self.create_strands_model(llm_type)
        
        # Get the appropriate prompt for the role
        if role_registry:
            # Use role-specific system prompt from role definition
            role_def = role_registry.get_role(role)
            if role_def:
                prompts = role_def.config.get('prompts', {})
                system_prompt = prompts.get('system', 'You are a helpful AI assistant.')
            else:
                system_prompt = self.prompt_library.get_prompt(role)
        else:
            # Fallback to prompt library
            system_prompt = self.prompt_library.get_prompt(role)
        
        # Create the agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools or []
        )
        
        # Cache the agent
        self._agent_cache[cache_key] = agent
        logger.info(f"Cached new agent for {cache_key}")
        
        return agent
    
    def _hash_tools(self, tools: List) -> str:
        """
        Create a hash of the tools list for caching purposes.
        
        Args:
            tools: List of tools
            
        Returns:
            Hash string representing the tools
        """
        if not tools:
            return "no_tools"
        
        # Create a stable hash based on tool names/types
        tool_names = []
        for tool in tools:
            if hasattr(tool, '__name__'):
                tool_names.append(tool.__name__)
            elif hasattr(tool, '_tool_name'):
                tool_names.append(tool._tool_name)
            else:
                tool_names.append(str(type(tool).__name__))
        
        # Sort for consistent hashing
        tool_names.sort()
        tools_str = "_".join(tool_names)
        
        return hashlib.md5(tools_str.encode()).hexdigest()[:8]
    
    def warm_models(self):
        """
        Pre-create commonly used models to avoid cold start delays.
        This should be called during application startup.
        """
        if self._is_warmed:
            logger.debug("Models already warmed, skipping")
            return
        
        logger.info("Warming up commonly used models...")
        
        try:
            # Pre-warm WEAK model for routing (most critical for performance)
            if LLMType.WEAK in self.configs:
                self.create_strands_model(LLMType.WEAK)
                logger.info("Warmed WEAK model for routing")
            
            # Pre-warm DEFAULT model for general tasks
            if LLMType.DEFAULT in self.configs:
                self.create_strands_model(LLMType.DEFAULT)
                logger.info("Warmed DEFAULT model for general tasks")
            
            # Pre-warm STRONG model if available
            if LLMType.STRONG in self.configs:
                self.create_strands_model(LLMType.STRONG)
                logger.info("Warmed STRONG model for complex tasks")
            
            self._is_warmed = True
            logger.info(f"Model warming completed. Cached {len(self._model_cache)} models")
            
        except Exception as e:
            logger.warning(f"Model warming failed: {e}")
            # Don't fail startup if warming fails
    
    def clear_cache(self):
        """
        Clear all cached models and agents.
        Useful for testing or memory management.
        """
        self._model_cache.clear()
        self._agent_cache.clear()
        self._is_warmed = False
        logger.info("Cleared all model and agent caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached models and agents.
        
        Returns:
            Dict with cache statistics
        """
        return {
            "models_cached": len(self._model_cache),
            "agents_cached": len(self._agent_cache),
            "is_warmed": self._is_warmed,
            "model_cache_keys": list(self._model_cache.keys()),
            "agent_cache_keys": list(self._agent_cache.keys())
        }

    def _get_config(self, llm_type: LLMType, name: Optional[str] = None) -> BaseConfig:
        """
        Get configuration for the specified LLM type and name.
        
        Args:
            llm_type: The LLM type
            name: Optional specific configuration name
            
        Returns:
            BaseConfig: The configuration object
            
        Raises:
            ValueError: If configuration not found
        """
        configs = self.configs.get(llm_type, [])
        if not configs:
            raise ValueError(f"No configurations found for LLMType '{llm_type}'")

        if name:
            config = next((c for c in configs if c.name == name), None)
            if not config:
                raise ValueError(f"No configuration found with name '{name}' for LLMType '{llm_type}'")
        else:
            config = configs[0]
        
        return config

    def get_framework(self) -> str:
        """Get the current framework being used (always 'strands')."""
        return self.framework

    def is_strands_available(self) -> bool:
        """Check if StrandsAgent is available."""
        return STRANDS_AVAILABLE

    def get_available_llm_types(self) -> List[LLMType]:
        """Get list of available LLM types with configurations."""
        return list(self.configs.keys())

    def get_config_count(self, llm_type: LLMType) -> int:
        """Get number of configurations for the specified LLM type."""
        return len(self.configs.get(llm_type, []))

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the factory configuration.
        
        Returns:
            Dict containing validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "llm_types": len(self.configs),
            "total_configs": sum(len(configs) for configs in self.configs.values()),
            "framework": self.framework,
            "strands_available": STRANDS_AVAILABLE
        }
        
        # Check if we have at least one configuration
        if not self.configs:
            validation_result["valid"] = False
            validation_result["errors"].append("No LLM configurations found")
        
        # Check each LLM type has valid configurations
        for llm_type, configs in self.configs.items():
            if not configs:
                validation_result["warnings"].append(f"No configurations for LLM type: {llm_type}")
            
            for config in configs:
                if not hasattr(config, 'provider_type') and not hasattr(config, 'provider_name'):
                    validation_result["errors"].append(f"Configuration missing provider type: {config}")
                    validation_result["valid"] = False
        
        return validation_result
    
    def get_agent(self, llm_type: LLMType, provider: str = None) -> Any:
        """
        Get cached Agent for provider/model combination.
        
        Args:
            llm_type: Semantic model type (WEAK, DEFAULT, STRONG)
            provider: Provider name (defaults to primary provider)
            
        Returns:
            Cached Agent instance ready for context switching
        """
        # Create pool key based on infrastructure, not business logic
        provider = provider or self._get_default_provider()
        pool_key = f"{provider}_{llm_type.value}"
        
        # Return cached Agent if available
        if pool_key in self._agent_pool:
            self._pool_stats['hits'] += 1
            logger.debug(f"⚡ Agent pool hit for {pool_key}")
            return self._agent_pool[pool_key]
        
        # Create new Agent with cached model
        self._pool_stats['misses'] += 1
        logger.info(f"🔧 Creating new Agent for {pool_key}")
        
        model = self.create_strands_model(llm_type)
        agent = Agent(model=model)  # Minimal Agent creation
        
        # Cache the Agent
        self._agent_pool[pool_key] = agent
        self._pool_stats['created'] += 1
        logger.info(f"✅ Cached new Agent for {pool_key}")
        
        return agent
    
    def warm_agent_pool(self):
        """Pre-warm Agent pool for common provider/model combinations."""
        logger.info("🔥 Warming Agent pool...")
        
        common_combinations = [
            (LLMType.WEAK, 'bedrock'),    # Fast routing and simple tasks
            (LLMType.DEFAULT, 'bedrock'), # Standard tasks
            (LLMType.STRONG, 'bedrock'),  # Complex planning
        ]
        
        for llm_type, provider in common_combinations:
            try:
                agent = self.get_agent(llm_type, provider)
                logger.info(f"✅ Pre-warmed {provider}_{llm_type.value}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to pre-warm {provider}_{llm_type.value}: {e}")
        
        logger.info(f"🔥 Agent pool warmed: {len(self._agent_pool)} agents ready")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get Agent pool performance statistics."""
        total_requests = self._pool_stats['hits'] + self._pool_stats['misses']
        hit_rate = (self._pool_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'pool_size': len(self._agent_pool),
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            **self._pool_stats
        }
    
    def _get_default_provider(self) -> str:
        """Get the default provider from available configurations."""
        # Return the first available provider type
        for llm_type, configs in self.configs.items():
            if configs:
                config = configs[0]
                if hasattr(config, 'provider_type'):
                    return config.provider_type
                elif hasattr(config, 'provider_name'):
                    return config.provider_name
        
        # Fallback to bedrock if no provider found
        return "bedrock"
    
    def clear_agent_pool(self):
        """
        Clear all cached agents in the pool.
        Useful for testing or memory management.
        """
        self._agent_pool.clear()
        self._pool_stats = {'hits': 0, 'misses': 0, 'created': 0}
        logger.info("Cleared agent pool and reset statistics")