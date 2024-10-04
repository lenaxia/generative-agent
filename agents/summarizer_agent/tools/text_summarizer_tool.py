import yaml
from typing import Any, Dict, Optional
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from llm_provider.base_client import BaseLLMClient
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_file_path = os.path.join(current_dir, "prompts.yaml")

# Load prompts from prompts.yaml
with open(prompts_file_path, "r") as f:

    prompts = yaml.safe_load(f)

class SummaryValidationOutput(BaseModel):
    factualness_score: float = Field(..., description="Score indicating the factualness of the summary")
    completeness_score: float = Field(..., description="Score indicating the completeness of the summary")

class TextSummarizerTool(BaseTool):
    name: str = "TextSummarizerTool"
    description: str = "This tool summarizes long texts and validates the summary for factualness and completeness."

    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    def _run(self, text: str, max_summary_length: int = 500) -> Dict[str, Any]:
        summary_prompt = PromptTemplate(
            template=prompts["summarize_prompt"],
            input_variables=["text", "max_summary_length"],
        )
        summary_chain = LLMChain(llm=self.llm.get_llm(), prompt=summary_prompt)
        summary = summary_chain.run(text=text, max_summary_length=max_summary_length)
    
        validation_prompt = PromptTemplate(
            template=prompts["validation_prompt"],
            input_variables=["summary", "text"],
        )
        validation_chain = LLMChain(llm=self.llm.get_llm(), prompt=validation_prompt)
    
        validation_output = validation_chain.run(summary=summary, text=text)
        validation_result = SummaryValidationOutput.parse_raw(validation_output)
    
        if validation_result.factualness_score < 0.8:
            # Retry summary if factualness score is too low
            return self._run(text, max_summary_length)
    
        return {
            "summary": summary,
            "factualness_score": validation_result.factualness_score,
            "completeness_score": validation_result.completeness_score,
        }
    
    def _arun(self, text: str, max_summary_length: int = 500) -> Dict[str, Any]:
        return self._run(text, max_summary_length)

    async def _arun_async(self, text: str, max_summary_length: int = 500) -> Dict[str, Any]:
        raise NotImplementedError("TestSummarizerTool does not support async mode")
