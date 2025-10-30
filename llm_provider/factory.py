"""LLM Factory module for creating and managing language model instances.

This module provides factory classes for creating different types of LLM
instances including OpenAI, Anthropic, and Bedrock providers with proper
configuration management and caching.
"""

import hashlib
import logging
from enum import Enum
from typing import Any

import boto3
from botocore.config import Config

from config.base_config import BaseConfig

# Import StrandsAgent dependencies with fallback for testing
try:
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    STRANDS_AVAILABLE = True
except ImportError:
    # Mock classes for testing when StrandsAgent is not available
    class BedrockModel:
        """Mock BedrockModel class for testing when StrandsAgent is not available.

        Provides a placeholder implementation for Bedrock model functionality
        during testing or when the StrandsAgent library is not installed.
        """

        def __init__(self, **kwargs):
            """Initialize mock BedrockModel with provided arguments.

            Args:
                **kwargs: Configuration arguments for the mock model.
            """
            self.kwargs = kwargs

    class Agent:
        """Mock Agent class for testing when StrandsAgent is not available.

        Provides a placeholder implementation for agent functionality
        during testing or when the StrandsAgent library is not installed.
        """

        def __init__(self, **kwargs):
            """Initialize mock Agent with provided arguments.

            Args:
                **kwargs: Configuration arguments for the mock agent.
            """
            self.kwargs = kwargs

        def __call__(self, instruction):
            """Execute the mock agent with the given instruction.

            Provides a simple mock response for testing purposes when
            the StrandsAgent library is not available.

            Args:
                instruction: The instruction or prompt to process.

            Returns:
                A mock response string indicating the instruction received.
            """
            return f"Mock agent response to: {instruction}"

    STRANDS_AVAILABLE = False

# Try to import OpenAI model separately since it might not be available
try:
    from strands.models.openai import OpenAIModel

    OPENAI_AVAILABLE = True
except ImportError:

    class OpenAIModel:
        """Mock OpenAIModel class for testing when OpenAI integration is not available.

        Provides a placeholder implementation for OpenAI model functionality
        during testing or when the OpenAI integration is not available.
        """

        def __init__(self, **kwargs):
            """Initialize mock OpenAIModel with provided arguments.

            Args:
                **kwargs: Configuration arguments for the mock model.
            """
            self.kwargs = kwargs

    OPENAI_AVAILABLE = False

from llm_provider.prompt_library import PromptLibrary

logger = logging.getLogger(__name__)

# Log import status after logger is defined
if STRANDS_AVAILABLE:
    logger.info("Successfully imported Strands components")
else:
    logger.warning("Strands import failed, using mock components")


class LLMType(Enum):
    """Enumeration of available LLM types and capabilities.

    Defines the different types of language models available in the system,
    categorized by their capabilities and intended use cases.
    """

    DEFAULT = "default"
    STRONG = "strong"
    WEAK = "weak"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    IMAGE_GENERATION = "image_generation"


class LLMFactory:
    """Factory class for creating and managing LLM instances.

    Provides centralized creation and configuration of different types of
    language models, including provider-specific implementations and
    fallback mechanisms.
    """

    def __init__(
        self, configs: dict[LLMType, list[BaseConfig]], framework: str = "strands"
    ):
        """Initialize LLMFactory with StrandsAgent support and performance caching.

        Args:
            configs: Configuration mapping for different LLM types
            framework: Framework to use (only "strands" supported)
        """
        self.configs = configs
        self.framework = "strands"  # Only StrandsAgent framework supported
        self.prompt_library = PromptLibrary()

        # Performance optimization: Model and agent caching
        self._model_cache: dict[str, Any] = {}
        self._agent_cache: dict[str, Any] = {}
        self._is_warmed = False

        # Enhanced caching infrastructure for agent pooling
        self._agent_pool: dict[str, Any] = {}  # Provider-based Agent pool
        self._pool_stats = {"hits": 0, "misses": 0, "created": 0}

        # Connection pooling optimization: Shared boto3 client for Bedrock
        self._bedrock_client = self._create_optimized_bedrock_client()

        logger.info(
            "LLMFactory initialized with caching, agent pooling, and optimized connection pooling enabled"
        )

    def add_config(self, llm_type: LLMType, config: BaseConfig):
        """Add a configuration for the specified LLM type."""
        if llm_type not in self.configs:
            self.configs[llm_type] = []
        self.configs[llm_type].append(config)

    def remove_config(self, llm_type: LLMType, name: str):
        """Remove a configuration by name for the specified LLM type."""
        if llm_type not in self.configs:
            return
        self.configs[llm_type] = [
            config for config in self.configs[llm_type] if config.name != name
        ]

    def create_strands_model(self, llm_type: LLMType, name: str | None = None):
        """Create a StrandsAgent model instance with caching for performance.

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

    def _create_new_model(self, llm_type: LLMType, name: str | None = None):
        """Internal method to create a new model instance.

        Args:
            llm_type: The semantic LLM type
            name: Optional specific configuration name

        Returns:
            New StrandsAgent model instance
        """
        config = self._get_config(llm_type, name)

        provider_type = self._extract_provider_type(config)
        model_params = self._extract_model_parameters(config)

        return self._create_model_instance(provider_type, model_params)

    def _extract_provider_type(self, config) -> str:
        """Extract provider type from configuration."""
        if hasattr(config, "provider_type"):
            return config.provider_type
        elif hasattr(config, "provider_name"):
            return config.provider_name
        else:
            raise ValueError("Configuration missing provider type information")

    def _extract_model_parameters(self, config) -> dict:
        """Extract model parameters from configuration."""
        model_id = self._extract_model_id(config)
        temperature = self._extract_temperature(config)

        model_params = {"model_id": model_id, "temperature": temperature}

        # Add additional parameters if available
        if hasattr(config, "additional_params") and config.additional_params:
            model_params.update(config.additional_params)

        return model_params

    def _extract_model_id(self, config) -> str:
        """Extract model_id from configuration."""
        if hasattr(config, "model_id") and config.model_id:
            return config.model_id
        elif (
            hasattr(config, "llm_config")
            and hasattr(config.llm_config, "model")
            and config.llm_config.model
        ):
            return config.llm_config.model
        elif (
            hasattr(config, "llm_config")
            and hasattr(config.llm_config, "model_id")
            and config.llm_config.model_id
        ):
            return config.llm_config.model_id
        else:
            raise ValueError(f"Configuration missing model_id: {config}")

    def _extract_temperature(self, config) -> float:
        """Extract temperature from configuration."""
        if hasattr(config, "temperature") and config.temperature:
            return config.temperature
        elif (
            hasattr(config, "llm_config")
            and hasattr(config.llm_config, "temperature")
            and config.llm_config.temperature
        ):
            return config.llm_config.temperature
        else:
            return 0.3  # default

    def _create_optimized_bedrock_client(self):
        """Create optimized boto3 Bedrock client with connection pooling.

        This client is shared across all BedrockModel instances to enable
        connection reuse and prevent idle connection timeouts.

        Returns:
            Configured boto3 bedrock-runtime client
        """
        try:
            # Extract region from Bedrock config
            region = self._get_bedrock_region()

            boto_config = Config(
                region_name=region,
                # Adaptive retry mode with intelligent backoff
                retries={"max_attempts": 3, "mode": "adaptive"},
                # Aggressive connection pooling to maintain warm connections
                max_pool_connections=50,
                # TCP keepalive to prevent connection timeouts
                tcp_keepalive=True,
                # Faster timeout detection for stale connections
                connect_timeout=5,
                read_timeout=60,
                # Performance optimization
                parameter_validation=False,
            )

            client = boto3.client("bedrock-runtime", config=boto_config)
            logger.info(
                f"✅ Optimized Bedrock client created with connection pooling (region: {region})"
            )
            return client

        except Exception as e:
            logger.warning(f"⚠️ Failed to create optimized Bedrock client: {e}")
            logger.warning("Falling back to default boto3 client configuration")
            # Use same region extraction for fallback
            region = self._get_bedrock_region()
            return boto3.client("bedrock-runtime", region_name=region)

    def _get_bedrock_region(self) -> str:
        """Extract Bedrock region from configuration.

        Returns:
            Region name from config, defaults to us-west-2
        """
        try:
            # Look for Bedrock configs in any LLM type
            for llm_type, configs in self.configs.items():
                for config in configs:
                    # Check if this is a Bedrock config
                    provider = self._extract_provider_type(config)
                    if provider == "bedrock":
                        # Try to get region from config
                        if hasattr(config, "llm_config") and hasattr(
                            config.llm_config, "region_name"
                        ):
                            return config.llm_config.region_name
                        elif hasattr(config, "region_name"):
                            return config.region_name
                        elif hasattr(config, "region"):
                            return config.region

            # Default fallback
            logger.debug("No Bedrock region found in config, using default: us-west-2")
            return "us-west-2"

        except Exception as e:
            logger.warning(f"Error extracting Bedrock region: {e}, using default")
            return "us-west-2"

    def _create_model_instance(self, provider_type: str, model_params: dict):
        """Create model instance based on provider type.

        For Bedrock, uses shared boto3 client with connection pooling.
        """
        if provider_type == "bedrock":
            return BedrockModel(client=self._bedrock_client, **model_params)
        elif provider_type == "openai":
            if OPENAI_AVAILABLE:
                return OpenAIModel(**model_params)
            else:
                raise ValueError(
                    "OpenAI model requested but not available in Strands installation"
                )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    def create_universal_agent(
        self,
        llm_type: LLMType,
        role: str,
        tools: list | None = None,
        role_registry=None,
    ):
        """Create a Universal Agent using StrandsAgent framework with caching.

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
                prompts = role_def.config.get("prompts", {})
                system_prompt = prompts.get("system", "You are a helpful AI assistant.")
            else:
                system_prompt = self.prompt_library.get_prompt(role)
        else:
            # Fallback to prompt library
            system_prompt = self.prompt_library.get_prompt(role)

        # Create the agent
        agent = Agent(model=model, system_prompt=system_prompt, tools=tools or [])

        # Cache the agent
        self._agent_cache[cache_key] = agent
        logger.info(f"Cached new agent for {cache_key}")

        return agent

    def _hash_tools(self, tools: list) -> str:
        """Create a hash of the tools list for caching purposes.

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
            if hasattr(tool, "__name__"):
                tool_names.append(tool.__name__)
            elif hasattr(tool, "_tool_name"):
                tool_names.append(tool._tool_name)
            else:
                tool_names.append(str(type(tool).__name__))

        # Sort for consistent hashing
        tool_names.sort()
        tools_str = "_".join(tool_names)

        return hashlib.md5(tools_str.encode()).hexdigest()[:8]

    def warm_models(self):
        """Pre-create commonly used models to avoid cold start delays.

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
            logger.info(
                f"Model warming completed. Cached {len(self._model_cache)} models"
            )

        except Exception as e:
            logger.warning(f"Model warming failed: {e}")
            # Don't fail startup if warming fails

    def clear_cache(self):
        """Clear all cached models and agents.

        Useful for testing or memory management.
        """
        self._model_cache.clear()
        self._agent_cache.clear()
        self._is_warmed = False
        logger.info("Cleared all model and agent caches")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about cached models and agents.

        Returns:
            Dict with cache statistics
        """
        return {
            "models_cached": len(self._model_cache),
            "agents_cached": len(self._agent_cache),
            "is_warmed": self._is_warmed,
            "model_cache_keys": list(self._model_cache.keys()),
            "agent_cache_keys": list(self._agent_cache.keys()),
        }

    def _get_config(self, llm_type: LLMType, name: str | None = None) -> BaseConfig:
        """Get configuration for the specified LLM type and name.

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
                raise ValueError(
                    f"No configuration found with name '{name}' for LLMType '{llm_type}'"
                )
        else:
            config = configs[0]

        return config

    def get_framework(self) -> str:
        """Get the current framework being used (always 'strands')."""
        return self.framework

    def is_strands_available(self) -> bool:
        """Check if StrandsAgent is available."""
        return STRANDS_AVAILABLE

    def get_available_llm_types(self) -> list[LLMType]:
        """Get list of available LLM types with configurations."""
        return list(self.configs.keys())

    def get_config_count(self, llm_type: LLMType) -> int:
        """Get number of configurations for the specified LLM type."""
        return len(self.configs.get(llm_type, []))

    def validate_configuration(self) -> dict[str, Any]:
        """Validate the factory configuration.

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
            "strands_available": STRANDS_AVAILABLE,
        }

        # Check if we have at least one configuration
        if not self.configs:
            validation_result["valid"] = False
            validation_result["errors"].append("No LLM configurations found")

        # Check each LLM type has valid configurations
        for llm_type, configs in self.configs.items():
            if not configs:
                validation_result["warnings"].append(
                    f"No configurations for LLM type: {llm_type}"
                )

            for config in configs:
                if not hasattr(config, "provider_type") and not hasattr(
                    config, "provider_name"
                ):
                    validation_result["errors"].append(
                        f"Configuration missing provider type: {config}"
                    )
                    validation_result["valid"] = False

        return validation_result

    def get_agent(self, llm_type: LLMType, provider: str = None) -> Any:
        """Get cached Agent for provider/model combination.

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
            self._pool_stats["hits"] += 1
            logger.debug(f"⚡ Agent pool hit for {pool_key}")
            return self._agent_pool[pool_key]

        # Create new Agent with cached model
        self._pool_stats["misses"] += 1
        logger.info(f"🔧 Creating new Agent for {pool_key}")

        model = self.create_strands_model(llm_type)
        agent = Agent(model=model)  # Minimal Agent creation

        # Cache the Agent
        self._agent_pool[pool_key] = agent
        self._pool_stats["created"] += 1
        logger.info(f"✅ Cached new Agent for {pool_key}")

        return agent

    def warm_agent_pool(self):
        """Pre-warm Agent pool for common provider/model combinations."""
        common_combinations = [
            (LLMType.WEAK, "bedrock"),  # Fast routing and simple tasks
            (LLMType.DEFAULT, "bedrock"),  # Standard tasks
            (LLMType.STRONG, "bedrock"),  # Complex planning
        ]

        for llm_type, provider in common_combinations:
            try:
                self.get_agent(llm_type, provider)
                logger.info(f"✅ Pre-warmed {provider}_{llm_type.value}")
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to pre-warm {provider}_{llm_type.value}: {e}"
                )

    def get_pool_stats(self) -> dict[str, Any]:
        """Get Agent pool performance statistics."""
        total_requests = self._pool_stats["hits"] + self._pool_stats["misses"]
        hit_rate = (
            (self._pool_stats["hits"] / total_requests * 100)
            if total_requests > 0
            else 0
        )

        return {
            "pool_size": len(self._agent_pool),
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            **self._pool_stats,
        }

    def _get_default_provider(self) -> str:
        """Get the default provider from available configurations."""
        # Return the first available provider type
        for _llm_type, configs in self.configs.items():
            if configs:
                config = configs[0]
                if hasattr(config, "provider_type"):
                    return config.provider_type
                elif hasattr(config, "provider_name"):
                    return config.provider_name

        # Fallback to bedrock if no provider found
        return "bedrock"

    def clear_agent_pool(self):
        """Clear all cached agents in the pool.

        Useful for testing or memory management.
        """
        self._agent_pool.clear()
        self._pool_stats = {"hits": 0, "misses": 0, "created": 0}
        logger.info("Cleared agent pool and reset statistics")
