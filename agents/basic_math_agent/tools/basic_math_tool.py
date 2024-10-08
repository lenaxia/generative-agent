import os
import yaml
from typing import Optional, Union, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, validator
import sympy
from llm_provider.base_client import BaseLLMClient

current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_file_path = os.path.join(current_dir, "prompts.yaml")

# Load prompts from prompts.yaml
with open(prompts_file_path, "r") as f:
    prompts = yaml.safe_load(f)

# Define input and output schemas
class BasicMathInput(BaseModel):
    expression: str = Field(..., description="The mathematical expression to evaluate.")

class AlgebraicStep(BaseModel):
    step: str
    explanation: Optional[str] = None

class AlgebraicResponse(BaseModel):
    steps: List[AlgebraicStep]
    result: Union[str, float, int]

class BasicMathOutput(BaseModel):
    result: Union[str, float, int] = Field(..., description="The result of the mathematical expression.")
    expression_type: str = Field(..., description="The type of the mathematical expression.")
    steps: Optional[List[str]] = Field(None, description="The steps to solve an algebraic expression.")

# Define the BasicMathTool
class BasicMathTool(BaseTool, BaseModel):
    name: str = "BasicMathTool"
    description: str = "This tool can perform basic arithmetic and algebraic operations."
    input_schema: Optional[BaseModel] = None
    output_schema: Optional[BaseModel] = None
    llm_provider: BaseLLMClient = None

    @validator("input_schema", pre=True, always=True)
    def validate_input(cls, input_schema):
        return BasicMathInput(expression="")
    
    @validator("output_schema", pre=True, always=True)
    def validate_output(cls, output_schema):
        return BasicMathOutput(result="", expression_type="")

    def __init__(self, llm_provider: BaseLLMClient):
        super().__init__()
        self.llm_provider = llm_provider

    def _run(self, input: BasicMathInput) -> BasicMathOutput:
        expression = input.expression
        self.llm_provider.initialize_responder(prompts["math_prompt"], [])

        try:
            # Try to evaluate the expression directly
            result = sympy.sympify(expression).evalf()
            return BasicMathOutput(result=result, expression_type="arithmetic")
        except sympy.SympifyError:
            # If the expression cannot be evaluated directly, use the LLM to break it down
            state = [prompts["math_prompt"].format(expression=expression)]
            response = self.llm_provider.respond(state)

            try:
                # Parse the LLM response into the AlgebraicResponse model
                algebraic_response = AlgebraicResponse.parse_raw(response["result"])

                return BasicMathOutput(
                    result=algebraic_response.result,
                    expression_type="algebraic",
                    steps=[step.step for step in algebraic_response.steps]
                )
            except (ValidationError, sympy.SympifyError, ValueError, TypeError) as e:
                raise ValueError(f"Invalid input or algebraic steps: {e}")

    def tool_schema(self) -> dict:
        return {
            "input": self.input_schema.schema(),
            "output": self.output_schema.schema(),
        }
