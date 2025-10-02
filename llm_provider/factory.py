from enum import Enum
from typing import Optional, Dict, List, Union, Type, Any
from config.base_config import BaseConfig

# Import StrandsAgent dependencies with fallback for testing
try:
    from strands.models import BedrockModel, OpenAIModel
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    # Mock classes for testing when StrandsAgent is not available
    class BedrockModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    
    class OpenAIModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    
    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    
    STRANDS_AVAILABLE = False

from pydantic import BaseModel
from llm_provider.prompt_library import PromptLibrary

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
        Initialize LLMFactory with StrandsAgent support only.
        
        Args:
            configs: Configuration mapping for different LLM types
            framework: Framework to use (only "strands" supported)
        """
        self.configs = configs
        self.framework = "strands"  # Only StrandsAgent framework supported
        self.prompt_library = PromptLibrary()

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
        Create a StrandsAgent model instance.
        
        Args:
            llm_type: The semantic LLM type (DEFAULT, STRONG, WEAK, etc.)
            name: Optional specific configuration name
            
        Returns:
            StrandsAgent model instance
            
        Raises:
            ValueError: If configuration not found or unsupported provider
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
            return OpenAIModel(**model_params)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    def create_universal_agent(self, llm_type: LLMType, role: str, tools: Optional[List] = None):
        """
        Create a Universal Agent using StrandsAgent framework.
        
        Args:
            llm_type: The semantic LLM type for model selection
            role: The agent role (determines system prompt)
            tools: Optional list of tools for the agent
            
        Returns:
            StrandsAgent Agent instance
        """
        # Create the model
        model = self.create_strands_model(llm_type)
        
        # Get the appropriate prompt for the role
        system_prompt = self.prompt_library.get_prompt(role)
        
        # Create and return the agent
        return Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools or []
        )

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