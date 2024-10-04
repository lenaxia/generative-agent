from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from langchain.tools import BaseTool
from llm_provider import BaseLLMClient
from llm_config import LLMRegistry, LLMType
from basic_math_tool import BasicMathTool, BasicMathInput, BasicMathOutput

class BasicMathAgent(BaseAgent):
    def __init__(self, llm_registry: LLMRegistry, message_bus: MessageBus, config: Optional[Dict] = None):
        super().__init__(llm_registry, message_bus, config)
        self.tool = BasicMathTool(llm_registry.get_llm(LLMType.DEFAULT))

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
