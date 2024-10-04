from logging import Logger
from typing import List, Dict, Optional
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from pydantic import parse_obj_as
from agents.base_agent import BaseAgent
from shared_tools.message_bus import MessageBus
from supervisor.llm_registry import LLMRegistry, LLMType
from supervisor.task_graph import TaskGraph
from llm_provider.base_client import BaseLLMClient

class PlanningAgent(BaseAgent):
    def __init__(self, logger: Logger, llm_registry: LLMRegistry, message_bus: MessageBus, agent_id: str, config: Optional[Dict] = None):
        super().__init__(logger, llm_registry, message_bus, agent_id, config)
        self.agents = []

    def set_agents(self, agents: List[BaseAgent]):
        """
        Sets the list of available agents for the PlanningAgent.
        """
        self.agents = agents
        self.tools = self.create_tools()

    def create_tools(self) -> List[BaseTool]:
        tools = []
        for agent in self.agents:
            agent_tools = agent.tools
            for tool_name, tool in agent_tools.items():
                tools.append(BaseTool(
                    name=f"{agent.__class__.__name__} - {tool.name}",
                    description=tool.description,
                    args_schema=tool.args_schema,
                    func=lambda x: f"Using {tool.name} from {agent.__class__.__name__} agent.",
                ))
        return tools

    @property
    def tools(self) -> Dict[str, BaseTool]:
        return {tool.name: tool for tool in self.tools}

    def _run(self, llm_client: BaseLLMClient, instruction: str, llm_type: LLMType = LLMType.DEFAULT) -> TaskGraph:
        prompt_template = PromptTemplate(
            input_variables=["input", "agents"],
            template="You are a planning agent responsible for breaking down complex tasks into a sequence of smaller subtasks. "
                     "Given the available agents and their capabilities, create a task graph to accomplish the following task: {input}\n\n"
                     "Agents:\n{agents}\n\n"
                     "Task Graph:",
            formatter={
                "agents": lambda agents: "\n".join([f"- {agent.__class__.__name__} ({agent.description})\n  Tools:\n    {' '.join([f'- {tool.name} ({tool.description})' for tool_name, tool in agent.tools.items()])}" for agent in agents])
            },
        )

        tools = self.tools.values()
        agent = AgentExecutor.from_tools_and_prompt(
            tools=list(tools),
            llm=llm_client.model,
            prompt=prompt_template,
        )

        response = agent({"input": instruction, "agents": self.agents}, return_intermediate_steps=True)

        task_graph = parse_obj_as(TaskGraph, response)
        return task_graph

    def _format_input(self, instruction: str, *args, **kwargs) -> str:
        return instruction

    def _process_output(self, task_graph: TaskGraph) -> TaskGraph:
        return task_graph
