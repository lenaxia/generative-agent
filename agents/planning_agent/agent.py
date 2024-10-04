from typing import List
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.agents.agent_toolkits import AgentToolkit
from langchain.prompts import PromptTemplate
from pydantic import parse_obj_as

class PlanningAgent(BaseAgent):
    def __init__(self, llm_registry: LLMRegistry, message_bus: MessageBus, agents: List[Agent]):
        super().__init__(llm_registry, message_bus)
        self.agents = agents
        self.tools = self.create_tools()

    def create_tools(self) -> List[BaseTool]:
        tools = []
        for agent in self.agents:
            for tool in agent.tools:
                tools.append(BaseTool(
                    name=f"{agent.name} - {tool.name}",
                    description=tool.description,
                    args_schema=tool.args_schema,
                    func=lambda x: f"Using {tool.name} from {agent.name} agent.",
                ))
        return tools

    @property
    def tools(self) -> Dict[str, BaseTool]:
        return {tool.name: tool for tool in self.tools}

    def _run(self, llm_client: BaseLLMClient, instruction: str) -> TaskGraph:
        toolkit = AgentToolkit(
            llm_client=llm_client.model,
            tools=self.tools,
            prompt_template=PromptTemplate(
                input_variables=["input", "agents"],
                template="You are a planning agent responsible for breaking down complex tasks into a sequence of smaller subtasks. "
                         "Given the available agents and their capabilities, create a task graph to accomplish the following task: {input}\n\n"
                         "Agents:\n{agents}\n\n"
                         "Task Graph:",
                formatter={
                    "agents": lambda agents: "\n".join([f"- {agent.name} ({agent.description})\n  Tools:\n    {' '.join([f'- {tool.name} ({tool.description})' for tool in agent.tools])}" for agent in agents])
                },
            ),
        )

        executor = AgentExecutor.from_agent_toolkit(toolkit)
        response = executor({"input": instruction, "agents": self.agents}, return_intermediate_steps=True)

        task_graph = parse_obj_as(TaskGraph, response)
        return task_graph

    def _format_input(self, instruction: str, *args, **kwargs) -> str:
        return instruction

    def _process_output(self, task_graph: TaskGraph) -> TaskGraph:
        return task_graph
