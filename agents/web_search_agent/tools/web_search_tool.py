import yaml
from typing import Any, Dict, List, Optional
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.pydantic_utils import BaseModel
from pydantic import Field, validator
from llm_provider import BaseLLMClient
from llm_config import LLMRegistry, LLMType

# Load prompts from the YAML file
with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

# Pydantic models for input and output
class WebSearchInput(BaseModel):
    query: str
    num_results: int = Field(5, ge=1, le=10, description="Number of search results to retrieve (between 1 and 10).")

    @validator('query')
    def validate_query(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be empty or contain only whitespace characters.")
        return value.strip()

class WebSearchResult(BaseModel):
    title: str
    url: str
    description: str
    relevance_score: int = Field(..., ge=1, le=5, description="Relevance score between 1 (least relevant) and 5 (most relevant).")

class WebSearchOutput(BaseModel):
    results: List[WebSearchResult]

# WebSearchTool class
class WebSearchTool(BaseTool):
    name = "web_search_tool"
    description = "A tool for performing web searches and retrieving the top relevant results."
    input_schema: Optional[BaseModel] = WebSearchInput
    output_schema: Optional[BaseModel] = WebSearchOutput

    def __init__(self, llm_client: BaseLLMClient):
        self.query_prompt = prompts["web_search_tool"]["query_prompt"]
        self.result_format = prompts["web_search_tool"]["result_format"]
        self.llm_client = llm_client

        prompt_template = ChatPromptTemplate(
            template=self.query_prompt,
            input_variables=["query", "num_results"]
        )
        self.llm_client.initialize_responder(prompt_template, tools=[WebSearchOutput])

    def _run(self, input: WebSearchInput) -> WebSearchOutput:
        state = [input.dict()]
        response = self.llm_client.respond(state)

        results = []
        for idx, result_str in enumerate(response.split("\n\n")):
            if result_str.strip():
                result_data = result_str.format(result_idx=idx + 1)
                result = WebSearchResult.parse_raw(result_data)
                results.append(result)

        return WebSearchOutput(results=results)

    def _arun(self, input: WebSearchInput) -> WebSearchOutput:
        raise NotImplementedError("This tool does not support async execution.")

    def tool_schema(self) -> dict:
        return {
            "input": self.input_schema.schema(),
            "output": self.output_schema.schema(),
        }
