from enum import Enum
from typing import Optional
from llm_provider import BaseLLMClient

class LLMType(Enum):
    DEFAULT = 'default'
    STRONG = 'strong'
    WEAK = 'weak'
    VISION = 'vision'
    MULTIMODAL = 'multimodal'
    IMAGE_GENERATION = 'image_generation'

class LLMRegistry:
    def __init__(self, default_llm: BaseLLMClient, strong_llm: Optional[BaseLLMClient] = None, weak_llm: Optional[BaseLLMClient] = None):
        self.llms = {
            LLMType.DEFAULT: default_llm,
            LLMType.STRONG: strong_llm,
            LLMType.WEAK: weak_llm
        }

    def add_llm(self, llm_type: LLMType, llm_client: BaseLLMClient):
        """
        Adds a new LLM client to the config.
        """
        self.llms[llm_type] = llm_client

    def get_llm(self, llm_type: LLMType) -> Optional[BaseLLMClient]:
        """
        Retrieves the LLM client based on the specified type.
        """
        return self.llms.get(llm_type)
