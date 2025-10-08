from typing import Annotated, Dict, List
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from agents.base_agent import BaseAgent
from common.message_bus import MessageBus
from llm_provider.factory import LLMFactory, LLMType
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_community.retrievers import BM25Retriever
import logging
import subprocess
import sys
import time
import traceback

logger = logging.getLogger(__name__)

class WritePython(BaseModel):
    reasoning: str = Field(..., description="Conceptual solution.")
    pseudocode: str = Field(..., description="Detailed English pseudocode.")
    code: str = Field(..., description="Valid Python 3 solution to the problem")

class SolverInput(BaseModel):
    problem_description: str = Field(..., description="Description of the programming problem to be solved.")
    test_cases: List[Dict[str, str]] = Field(..., description="List of test cases for the problem.")
    runtime_limit: int = Field(..., description="Maximum runtime limit for the solution in seconds.")

class SolverOutput(BaseModel):
    description: str = Field(..., description="Detailed English description of the solution.")
    code: str = Field(..., description="Python code solution for the given problem. This should ONLY contain code")

class State(Dict):
    problem_description: str
    candidate: Annotated[AIMessage, add_messages]
    examples: str
    messages: Annotated[list[AnyMessage], add_messages]
    test_cases: List[Dict[str, str]]
    runtime_limit: int
    status: str

class TestCase(Dict):
    inputs: str
    outputs: str

class CodingAgent(BaseAgent):
    def __init__(self, logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.agent_description = "Solve programming problems by generating Python code solutions."
        self.config = config or {}
        self.logger = logger
        self.initialize()

    def _select_llm_provider(self):
        llm = self.llm_factory.create_chat_model(self.config.get("llm_class", LLMType.DEFAULT))
        return llm

    def _run(self, input: SolverInput) -> SolverOutput:
        self.logger.info(f"Running SolverAgent with problem description: {input.problem_description}")

        test_cases_str = "\n".join([f"Input: {tc['input']}\nExpected Output: {tc['output']}" for tc in input.test_cases])

        config = {"configurable": {"thread_id": "silver-hl-1", "k": 2}}

        solver_output = self.graph.invoke({"problem_description": input.problem_description, "test_cases": test_cases_str, "runtime_limit": input.runtime_limit}, config)

        return solver_output

    def _format_input(self, task_data: Dict) -> SolverInput:
        """Formats the input for the LLM provider and the respective tool(s).
        """
        input: SolverInput = SolverInput(**task_data)
        return input

    def _process_output(self, task_data: Dict, output: SolverOutput) -> Dict:
        return {"code": output.code}

    def run(self, task_data: Dict) -> Dict:
        self.setup()
        input = self._format_input(task_data)
        solver_output = self._run(input)
        self.teardown()
        return self._process_output(task_data, solver_output)

    def setup(self):
        self.retriever = BM25Retriever.from_texts(["These are some examples"])

        self.builder = StateGraph(State)

        draft_parser = PydanticOutputParser(pydantic_object=WritePython)

        draft_prompt = PromptTemplate(
            input_variables=["problem_description", "test_cases", "runtime_limit"],
            template="Given the following programming problem description and test cases, provide a draft Python code solution:\n\n"
                    "Problem Description: {problem_description}\n\n"
                    "Test Cases:\n{test_cases}\n\n"
                    "Format Instructions:\n{format_instructions}\n\n"
                    "Draft Solution (Do not provide additional commentary, only respond with the JSON):\n",
            partial_variables={"format_instructions": draft_parser.get_format_instructions()},
        )

        draft_chain = draft_prompt | self._select_llm_provider() | draft_parser


        solver_parser = PydanticOutputParser(pydantic_object=SolverOutput)

        solve_prompt = PromptTemplate(
            input_variables=["problem_description", "test_cases", "runtime_limit", "examples"],
            template="Given the following programming problem description, test cases, runtime limit, and examples of previously solved problems, provide a Python code solution:\n\n"
                    "Problem Description: {problem_description}\n\n"
                    "Test Cases:\n{test_cases}\n\n"
                    "Runtime Limit: {runtime_limit} seconds\n\n"
                    "Format Instructions:\n{format_instructions}\n\n"
                    "Examples of Previously Solved Problems:\n{examples}\n\n"
                    "Solution (Only provide the code):\n",
            partial_variables={"format_instructions": solver_parser.get_format_instructions()},
        )

        solve_chain = solve_prompt | self._select_llm_provider() | solver_parser


        self.builder.add_node("draft", draft_chain)
        self.builder.add_node("retrieve", self.retrieve_examples)
        self.builder.add_node("solve", solve_chain)
        self.builder.add_node("evaluate", self.evaluate)

        self.builder.add_edge(START, "draft")
        self.builder.add_edge("draft", "evaluate")
        self.builder.add_edge("draft", "retrieve")
        self.builder.add_edge("retrieve", "solve")
        self.builder.add_edge("solve", "evaluate")

        def control_edge(state: State):
            if state.get("status") == "success":
                return END
            return "solve"

        self.builder.add_conditional_edges("evaluate", control_edge, {END: END, "solve": "solve"})
        self.checkpointer = MemorySaver()
        self.graph = self.builder.compile(checkpointer=self.checkpointer)

    def teardown(self):
        pass

    def format_example(self, row):
        question = row["description"]
        answer = row["solution"]
        return f"""<problem>
{question}
</problem>
<solution>
{answer}
</solution>"""

    def retrieve_examples(self, state: State, config: RunnableConfig):
        top_k = config["configurable"].get("k") or 2
        ai_message: AIMessage = state["candidate"]
        if not ai_message.tool_calls:
            raise ValueError("Draft agent did not produce a valid code block")
        code = ai_message.tool_calls[0]["args"]["code"]
        examples_str = "\n".join(
            [doc.page_content for doc in self.retriever.invoke(code)[:top_k]]
        )
        examples_str = f"""
You previously solved the following problems in this competition:
<Examples>
{examples_str}
<Examples>
Approach this new question with similar sophistication."""
        return {"examples": examples_str}



    def format_tool_message(self, response: str, ai_message: AIMessage):
        return ToolMessage(
            content=response + "\nMake all fixes using the writePython tool.",
            tool_call_id=ai_message.tool_calls[0]["id"],
        )

    def check_correctness(self, code: str, input_data: str, expected_output: str, timeout: float) -> str:
        """Function to execute the generated code and check its correctness against the test cases.
        """
        try:
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            if time.time() - start_time > timeout:
                raise TimeoutError("Execution timed out.")
            if process.returncode != 0:
                return f"failed: {stderr}"
            else:
                if stdout.strip() == expected_output.strip():
                    return "passed"
                else:
                    return f"wrong answer. Expected '{expected_output}', got '{stdout}'"
        except subprocess.TimeoutExpired:
            process.kill()
            return "timed out"
        except Exception:
            return f"failed: {traceback.format_exc()}"

    def evaluate(self, state: State) -> Dict:
        """Node function to evaluate the generated solution.
        """
        test_cases = state["test_cases"]
        ai_message: AIMessage = state["messages"][-1]
        if not ai_message.tool_calls:
            return {
                "messages": [
                    HumanMessage(
                        content="No code submitted. Please try again using the correct python code."
                    )
                ]
            }
        try:
            code = ai_message.tool_calls[0]["args"]["code"]
        except Exception as e:
            return {"messages": [self.format_tool_message(repr(e), ai_message)]}
        num_test_cases = len(test_cases)
        succeeded = 0
        test_results = []
        for test_case in test_cases:
            input_data = test_case["input"]
            expected_output = test_case["output"]
            test_result = self.check_correctness(code, input_data, expected_output, state["runtime_limit"])
            test_results.append(test_result)
            if test_result == "passed":
                succeeded += 1
        pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
        if pass_rate == 1:
            return {"status": "success"}

        responses = "\n".join(
            [f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)]
        )
        response = f"Incorrect submission. Please respond with updated code.\nPass rate: {succeeded}/{num_test_cases}\nResults:\n{responses}"
        formatted_message = self.format_tool_message(response, ai_message)
        return {"messages": [formatted_message]}
