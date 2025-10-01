from enum import Enum
from typing import Optional, Dict, List, Union, Type
from config.base_config import BaseConfig

# Import LangChain dependencies with fallback for testing
try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain_aws import ChatBedrock
    from langchain_core.output_parsers import BaseOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for testing environments
    ChatPromptTemplate = None
    BaseTool = None
    ChatOpenAI = None
    ChatBedrock = None
    BaseOutputParser = None
    LANGCHAIN_AVAILABLE = False

# Import StrandsAgent dependencies with fallback
try:
    from strands.models import BedrockModel, OpenAIModel
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    # Mock classes for testing
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
        Initialize LLMFactory with enhanced StrandsAgent support.
        
        Args:
            configs: Configuration mapping for different LLM types
            framework: Framework to use ("strands" or "langchain")
        """
        self.configs = configs
        self.framework = framework if framework in ["strands", "langchain"] else "strands"
        self.prompt_library = PromptLibrary()

    def add_config(self, llm_type: LLMType, config: BaseConfig):
        if llm_type not in self.configs:
            self.configs[llm_type] = []
        self.configs[llm_type].append(config)

    def remove_config(self, llm_type: LLMType, name: str):
        if llm_type not in self.configs:
            return
        self.configs[llm_type] = [config for config in self.configs[llm_type] if config.name != name]

    def create_provider(self, llm_type: LLMType, name: Optional[str] = None, tools: Optional[List[Union[Type[BaseModel], BaseTool]]] = None, prompt_template: Optional[ChatPromptTemplate] = None, output_parser: Optional[BaseOutputParser] = None):
        configs = self.configs.get(llm_type, [])
        if not configs:
            raise ValueError(f"No configurations found for LLMType '{llm_type}'")

        if name:
            config = next((c for c in configs if c.name == name), None)
            if not config:
                raise ValueError(f"No configuration found with name '{name}' for LLMType '{llm_type}'")
        else:
            config = configs[0]

        llm_type = config.provider_name
        llm_classes = {
            'openai': ChatOpenAI,
            'bedrock': ChatBedrock
        }

        if llm_type not in llm_classes:
            raise ValueError(f"Unknown model type: {llm_type}")

        llm_class = llm_classes[llm_type]
        llm_config = config.llm_config.dict(exclude_unset=True)

        model = llm_class(**llm_config)

        if tools:
            model = model.bind_tools(tools=tools)

        if prompt_template:
            model = prompt_template | model

        if output_parser:
            model = model | output_parser

        return model

    def create_chat_model(self, llm_type: LLMType, name: Optional[str] = None):
        if not isinstance(llm_type, LLMType):
            llm_type = LLMType.DEFAULT

        configs = self.configs.get(llm_type, [])
        if not configs:
            raise ValueError(f"No configurations found for LLMType '{llm_type}'")

        if name:
            config = next((c for c in configs if c.name == name), None)
            if not config:
                raise ValueError(f"No configuration found with name '{name}' for LLMType '{llm_type}'")
        else:
            config = configs[0]

        provider_type = config.get("type", None)
        provider_classes = {
            'openai': ChatOpenAI,
            'bedrock': ChatBedrock
        }

        if provider_type not in provider_classes:
            raise ValueError(f"Unknown provider type: {provider_type}")

        config = config.get("config", {})
        return provider_classes[provider_type](**config)

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
        # Note: We use mock classes when StrandsAgent is not available for testing
        
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
        # Note: We use mock classes when StrandsAgent is not available for testing
        
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
        """Get the current framework being used."""
        return self.framework

    def set_framework(self, framework: str):
        """
        Set the framework to use.
        
        Args:
            framework: "strands" or "langchain"
        """
        if framework in ["strands", "langchain"]:
            self.framework = framework
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def is_strands_available(self) -> bool:
        """Check if StrandsAgent is available."""
        return STRANDS_AVAILABLE

    def is_langchain_available(self) -> bool:
        """Check if LangChain is available."""
        return LANGCHAIN_AVAILABLE