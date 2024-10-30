from logging import Logger
from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from langchain.tools import BaseTool
from llm_provider.base_client import BaseLLMClient
from supervisor.llm_registry import LLMRegistry, LLMType
from agents.base_agent import BaseAgent
from .tools.basic_math_tool import BasicMathTool, BasicMathInput, BasicMathOutput
from common.message_bus import MessageBus

class BasicMathAgent(BaseAgent):
    # TODO: Needs refactor for the new agent design pattern using create_react_agent
    def __init__(self, logger: Logger, llm_registry: LLMRegistry, message_bus: MessageBus, agent_id: str, config: Optional[Dict] = None):
        super().__init__(logger, llm_registry, message_bus, agent_id, config)
        self.tool = BasicMathTool(llm_registry.get_client(LLMType.DEFAULT))

    @property
    def tools(self) -> Dict[str, BaseTool]:
        return {"math_tool": self.tool}

    def _run(self, llm_client: BaseLLMClient, *args, **kwargs) -> Any:
        
        expression = self._format_input(*args, **kwargs)
        math_input = BasicMathInput(expression=expression)
        math_output = self.tool._run(math_input)
        return self._process_output(math_output)

    def _arun(self, llm_client: BaseLLMClient, *args, **kwargs) -> Any:
        raise NotImplementedError("BasicMathAgent does not support async execution.")

    def _format_input(self, *args, **kwargs) -> str:
        expression = args[0]
        return expression

    def _process_output(self, output: BasicMathOutput) -> Dict[str, Any]:
        return {
            "result": output.result,
            "expression_type": output.expression_type,
            "steps": output.steps
        }

    def setup(self):
        # Perform any necessary setup operations here
        pass

    def teardown(self):
        # Perform any necessary teardown operations here
        pass
