from typing import TypedDict, Dict, Any, Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import Field
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel

from common.task_graph_v2 import TaskNode




class CodingState(TypedDict):
    prompt: str = Field(..., description="The original prompt")
    context: List[str] = Field(..., description="List of files relevant to the prompt")
    candidate: Any = Field(None, description="candidate solution")
    messages: Annotated[list[AnyMessage], add_messages]
    node: TaskNode
    language: str = Field("python", description="The language the agent is working on")
    status: str = Field("init", description="The current state of the agent")
    llm: BaseChatModel
    
class WriteCode(BaseModel):
    reasoning: str = Field(..., description="Conceptual solution.")
    pseudocode: str = Field(..., description="Detailed English pseudocode.")
    code: str = Field(..., description="Valid code solution to the problem in the language defined in the prompt")