import logging
from typing import Dict, Type, List, Union, Optional
from langchain_core.messages import ToolMessage
from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.language_models import BaseLanguageModel

from config.base_config import BaseConfig
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    pass

class BaseLLMClient(BaseModel):
    """
    Base class for LLM clients.

    Args:
        config (BaseConfig): The configuration object containing LLM provider settings.
        name (str): A unique name for the LLM client instance.
    """
    config: BaseConfig
    name: str
    model: Optional[BaseLanguageModel] = None
    tools: Optional[List[Union[Type[BaseModel], BaseTool]]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: BaseConfig, name: str, tools: Optional[List[Union[Type[BaseModel], BaseTool]]] = None):
        super().__init__(config=config, name=name)
        self.model = None
        self.tools = tools or []

    def initialize_responder(self, prompt_template: ChatPromptTemplate):
        """
        Initialize the responder for the LLM client.
        """
        if self.model is None:
            raise ValueError("LLM model is not initialized.")

        self.responder = ResponderWithRetries.from_llm(self.model, prompt_template, self.tools)

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
        tools (Optional[List[Union[Type[BaseModel], BaseTool]]]): A list of tools to be used by the LLM client.
    """
    def __init__(self, config: OpenAIConfig, name: str, tools: Optional[List[Union[Type[BaseModel], BaseTool]]] = None):
        super().__init__(config, name, tools)
        model_kwargs = {
            "stream": config.streaming,
            "request_kwargs": config.request_kwargs,
        }
        self.model = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            n=config.n,
            logprobs=config.logprobs,
            top_logprobs=config.top_logprobs,
            model_kwargs=model_kwargs,
            api_key=config.api_key,
        )

class ResponderWithRetries:
    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt_template: ChatPromptTemplate, tools: List[Union[Type[BaseModel], BaseTool]]):
        chain = prompt_template | llm.bind_tools(tools=tools)
        validation_tools = [tool.tool_call_schema if isinstance(tool, BaseTool) else tool for tool in tools]
        validator = PydanticToolsParser(tools=validation_tools)
        return cls(chain, validator)

    def __init__(self, chain, validator):
        self.chain = chain
        self.validator = validator

    def respond(self, state: list):
        response = []
        for attempt in range(3):
            response = self.chain.invoke(
                {"messages": state}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return response
            except ValidationError as e:
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
