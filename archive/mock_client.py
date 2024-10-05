from typing import Dict, List, Union
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel
from llm_provider import BaseLLMClient, ResponderWithRetries
from config import BaseConfig

class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing purposes.

    Args:
        config (BaseConfig): The configuration object containing LLM settings.
        name (str): A unique name for the LLM client instance.
        mock_responses (Dict[str, Any]): A dictionary mapping prompts to mock responses.
    """
    def __init__(self, config: BaseConfig, name: str, mock_responses: Dict[str, Any]):
        self.mock_responses = mock_responses
        super().__init__(config, name)

    def initialize_responder(self, prompt_template: ChatPromptTemplate, tools: List[Union[Type[BaseModel], BaseTool]]):
        """
        Initialize the responder for the Mock LLM client.

        Args:
            prompt_template (ChatPromptTemplate): The prompt template to use.
            tools (List[Union[Type[BaseModel], BaseTool]]): A list of tools or Pydantic models to use.
        """
        validator = PydanticToolsParser(tools=[tool.tool_call_schema if isinstance(tool, BaseTool) else tool for tool in tools])
        self.responder = MockResponderWithRetries(validator, self.mock_responses)

    def respond(self, state: list):
        """
        Invoke the Mock LLM client with the given state and return the mock response.

        Args:
            state (list): The initial state to use.

        Returns:
            Any: The mock response.
        """
        return self.responder.respond(state)

class MockResponderWithRetries(ResponderWithRetries):
    """
    Helper class for providing mock responses for LLM clients.

    Args:
        validator (PydanticToolsParser): The validator instance for validating responses.
        mock_responses (Dict[str, Any]): A dictionary mapping prompts to mock responses.
    """
    def __init__(self, validator, mock_responses):
        self.validator = validator
        self.mock_responses = mock_responses

    def respond(self, state: list):
        """
        Provide a mock response based on the given state.

        Args:
            state (list): The initial state to use.

        Returns:
            Any: The mock response.
        """
        prompt = " ".join(message.content for message in state[::-1])
        response = self.mock_responses.get(prompt, "Default mock response")
        try:
            self.validator.invoke(response)
            return response
        except ValidationError as e:
            raise ValueError(f"Mock response failed validation: {e}")
