from logging import Logger
from typing import Any, Dict
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from .tools.text_summarizer_tool import TextSummarizerTool
from supervisor.llm_registry import LLMRegistry, LLMType
from llm_provider.base_client import BaseLLMClient
from shared_tools.message_bus import MessageBus, MessageType
from agents.base_agent import BaseAgent


class TextSummarizeInput(BaseModel):
    text: str
    max_summary_length: int = Field(500, description="Maximum length of the summary")

class TextSummarizeOutput(BaseModel):
    summary: str
    factualness_score: float
    completeness_score: float

class TextSummarizerAgent(BaseAgent):
    def __init__(self, logger: Logger, llm_registry: LLMRegistry, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_registry, message_bus, agent_id, config)

    @property
    def tools(self) -> Dict[str, BaseTool]:
        llm_client = self.llm_registry.get_client(LLMType.DEFAULT)
        return {"text_summarizer": TextSummarizerTool(llm_client)}

    def _run(self, llm_client: BaseLLMClient, input_data: TextSummarizeInput = None) -> TextSummarizeOutput:
        tool = self.tools["text_summarizer"]

        input_dict = input_data.dict() if input_data else {}
        output = tool._run(**input_dict)

        return TextSummarizeOutput(**output)

    def _arun(self, llm_client: BaseLLMClient, input_data: TextSummarizeInput = None) -> TextSummarizeOutput:
        raise NotImplementedError("TextSummarizerAgent does not support async execution.")

    def _format_input(self, instruction: str, text: str, max_summary_length: int = 500) -> TextSummarizeInput:
        return TextSummarizeInput(text=text, max_summary_length=max_summary_length)

    def _process_output(self, output: TextSummarizeOutput) -> str:
        return f"Summary: {output.summary}\nFactualness Score: {output.factualness_score}\nCompleteness Score: {output.completeness_score}"

    def setup(self):
        pass

    def teardown(self):
        pass

    def handle_task_assignment(self, task_data: Dict):
        try:
            task_id = task_data["task_id"]
            agent_id = task_data["agent_id"]
            task_type = task_data["task_type"]
            text = task_data["text"]
            max_summary_length = task_data.get("max_summary_length", 500)
            request_id = task_data["request_id"]
            llm_client: BaseLLMClient = task_data["llm_client"]

            # Prepare input data
            input_data = self._format_input(task_type, text, max_summary_length)

            # Use the provided llm_client instance to respond to the task
            result = self._run(llm_client, input_data)

            # Publish the task response on the MessageBus
            response_data = {
                "task_id": task_id,
                "agent_id": agent_id,
                "task_type": task_type,
                "result": self._process_output(result),
                "request_id": request_id,
            }
            self.message_bus.publish(self, MessageType.TASK_RESPONSE, response_data)
        except Exception as e:
            logger.error(f"Summarizer: Error handling task assignment for task '{task_id}': {e}")
            # Publish an error message on the MessageBus
            error_data = {
                "request_id": request_id,
                "task_id": task_id,
                "error_message": str(e),
            }
            self.message_bus.publish(self, MessageType.AGENT_ERROR, error_data)
