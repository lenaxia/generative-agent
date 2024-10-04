import logging
from typing import Dict, Type, List, Union
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import ValidationError
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel

from config import AnthropicConfig, OpenAIConfig, ChatBedrockConfig

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    pass

class BaseLLMClient:
    """
    Base class for LLM clients.

    Args:
        config (BaseConfig): The configuration object containing LLM provider settings.
        name (str): A unique name for the LLM client instance.
    """
    def __init__(self, config, name: str):
        self.config = config
        self.name = name

    def initialize_responder(self, prompt_template: ChatPromptTemplate, tools: List[Union[Type[BaseModel], BaseTool]]):
        """
        Initialize the responder for the LLM client.

        Args:
            prompt_template (ChatPromptTemplate): The prompt template to use.
            tools (List[Union[Type[BaseModel], BaseTool]]): A list of tools or Pydantic models to use.
        """
        chain = prompt_template | self.model.bind_tools(tools=tools)
        validation_tools = [tool.tool_call_schema if isinstance(tool, BaseTool) else tool for tool in tools]
        validator = PydanticToolsParser(tools=validation_tools)
        self.responder = ResponderWithRetries(chain, validator)

    def respond(self, state: list):
        """
        Invoke the LLM client with the given state and return the response.

        Args:
            state (list): The initial state to use.

        Returns:
            Any: The response from the LLM client.
        """
        try:
            return self.responder.respond(state)
        except Exception as e:
            logger.error(f"Error in LLM client '{self.name}': {e}")
            raise LLMClientError(f"Error in LLM client '{self.name}': {e}") from e

class OpenAILLMClient(BaseLLMClient):
    """
    LLM client for OpenAI models.

    Args:
        config (OpenAIConfig): The configuration object containing OpenAI settings.
        name (str): A unique name for the LLM client instance.
    """
    def __init__(self, config: OpenAIConfig, name: str):
        super().__init__(config, name)
        self.model = ChatOpenAI(config=config)

class AnthropicLLMClient(BaseLLMClient):
    """
    LLM client for Anthropic models.

    Args:
        config (AnthropicConfig): The configuration object containing Anthropic settings.
        name (str): A unique name for the LLM client instance.
    """
    def __init__(self, config: AnthropicConfig, name: str):
        super().__init__(config, name)
        self.model = ChatAnthropic(config=config)

class BedrockLLMClient(BaseLLMClient):
    """
    LLM client for Bedrock models.

    Args:
        config (ChatBedrockConfig): The configuration object containing Bedrock settings.
        name (str): A unique name for the LLM client instance.
    """
    def __init__(self, config: ChatBedrockConfig, name: str):
        super().__init__(config, name)
        self.model = ChatBedrock(config=config)

class ResponderWithRetries:
    """
    Helper class for handling retries and validation of LLM responses.

    Args:
        chain (Any): The bound LLM chain.
        validator (PydanticToolsParser): The validator instance for validating responses.
    """
    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt_template: ChatPromptTemplate, tools: List[Union[Type[BaseModel], BaseTool]]):
        """
        Create a new instance of ResponderWithRetries from an LLM instance.

        Args:
            llm (BaseLanguageModel): The underlying LLM model instance.
            prompt_template (ChatPromptTemplate): The prompt template to use.
            tools (List[Union[Type[BaseModel], BaseTool]]): A list of tools or Pydantic models to use.

        Returns:
            ResponderWithRetries: A new instance of ResponderWithRetries.
        """
        chain = prompt_template | llm.bind_tools(tools=tools)
        validation_tools = [tool.tool_call_schema if isinstance(tool, BaseTool) else tool for tool in tools]
        validator = PydanticToolsParser(tools=validation_tools)
        return cls(chain, validator)

    def __init__(self, chain, validator):
        self.chain = chain
        self.validator = validator

    def respond(self, state: list):
        """
        Invoke the LLM chain with the given state and handle retries and validation.

        Args:
            state (list): The initial state to use.

        Returns:
            Any: The response from the LLM chain.
        """
        response = []
        num_retries = 0
        max_retries = 3

        while num_retries < max_retries:
            response = self.chain.invoke(
                {"messages": state}, {"tags": [f"attempt:{num_retries}"]}
            )
            try:
                self.validator.invoke(response)
                logger.info(f"LLM response validated successfully after {num_retries} retries.")
                return response
            except ValidationError as e:
                num_retries += 1
                logger.warning(f"Validation error in LLM response: {e}. Retrying ({num_retries}/{max_retries}).")
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return response


