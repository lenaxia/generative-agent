from enum import Enum
from typing import Optional, Dict
from llm_provider.base_client import BaseLLMClient
from pydantic import BaseModel, Field

class LLMType(Enum):
    DEFAULT = 'default'
    STRONG = 'strong'
    WEAK = 'weak'
    VISION = 'vision'
    MULTIMODAL = 'multimodal'
    IMAGE_GENERATION = 'image_generation'

class LLMRegistry(BaseModel):
    llm_clients: Dict[LLMType, Optional[BaseLLMClient]] = Field(default_factory=dict)

    def register_client(self, llm_client: BaseLLMClient, llm_type: LLMType):
        """
        Registers an LLM client with the registry.
        """
        self.llm_clients[llm_type] = llm_client

    def get_client(self, llm_type: LLMType) -> Optional[BaseLLMClient]:
        """
        Retrieves the LLM client based on the specified type.
        """
        return self.llm_clients.get(llm_type)
