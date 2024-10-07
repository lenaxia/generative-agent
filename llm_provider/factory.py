from enum import Enum
from typing import Optional, Dict, List, Union, Type
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel
#from langchain_openai import ChatOpenAI
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
    # TODO: Make this LLM agnostic, so it can support OpenAI (high priority), Bedrock (high priority) and Anthropic (low priority)
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
    
        #model = ChatOpenAI(
        #    base_url=config.endpoint,
        #    model_name=config.model_name,
        #    temperature=config.temperature,
        #    max_tokens=config.max_tokens,
        #    top_p=config.top_p,
        #    frequency_penalty=config.frequency_penalty,
        #    presence_penalty=config.presence_penalty,
        #    n=config.n,
        #    logprobs=config.logprobs,
        #    top_logprobs=config.top_logprobs,
        #    api_key=config.api_key,
        #)
        model = ChatBedrock(
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    model_kwargs=dict(temperature=0)
                )
    
        if tools:
            model = model.bind_tools(tools=tools)
    
        if prompt_template:
            model = prompt_template | model
    
        if output_parser:
            model = model | output_parser
    
        return model

    def create_chat_model(self, llm_type: LLMType, name: Optional[str] = None):
        configs = self.configs.get(llm_type, [])
        if not configs:
            raise ValueError(f"No configurations found for LLMType '{llm_type}'")
    
        if name:
            config = next((c for c in configs if c.name == name), None)
            if not config:
                raise ValueError(f"No configuration found with name '{name}' for LLMType '{llm_type}'")
        else:
            config = configs[0]
    
        #return ChatOpenAI(
        #    base_url=config.endpoint,
        #    model_name=config.model_name,
        #    temperature=config.temperature,
        #    max_tokens=config.max_tokens,
        #    top_p=config.top_p,
        #    frequency_penalty=config.frequency_penalty,
        #    presence_penalty=config.presence_penalty,
        #    n=config.n,
        #    logprobs=config.logprobs,
        #    top_logprobs=config.top_logprobs,
        #    api_key=config.api_key,
        #)
        return ChatBedrock(
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    model_kwargs=dict(temperature=0)
                )
    
