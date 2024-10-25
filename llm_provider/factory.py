from enum import Enum
from typing import Optional, Dict, List, Union, Type
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import BaseOutputParser
from config.base_config import BaseConfig

class LLMType(Enum):
    DEFAULT = 'default'
    STRONG = 'strong'
    WEAK = 'weak'
    VISION = 'vision'
    MULTIMODAL = 'multimodal'
    IMAGE_GENERATION = 'image_generation'

class LLMFactory:
    def __init__(self, configs: Dict[LLMType, List[BaseConfig]]):
        self.configs = configs

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