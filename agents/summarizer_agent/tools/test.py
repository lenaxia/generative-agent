import unittest.mock
from unittest.mock import MagicMock, patch
import pytest
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .text_summarizer_tool import TextSummarizerTool, SummaryValidationOutput

# Mock LLM client
class MockLLMClient:
    def get_client(self):
        return self

    def run(self, *args, **kwargs):
        return "This is a mock summary."

# Test data
INPUT_TEXT = "This is a long input text for testing purposes."
EXPECTED_SUMMARY = "This is a mock summary."
EXPECTED_FACTUALNESS_SCORE = 0.9
EXPECTED_COMPLETENESS_SCORE = 0.8

# Mock prompt templates
MOCK_SUMMARIZE_PROMPT = PromptTemplate(template="Mock summarization prompt.")
MOCK_VALIDATION_PROMPT = PromptTemplate(template="Mock validation prompt.")

@pytest.fixture
def mock_llm_client():
    return MockLLMClient()

@pytest.fixture
def text_summarizer_tool(mock_llm_client):
    with patch("text_summarizer_tool.prompts", {"summarize_prompt": MOCK_SUMMARIZE_PROMPT, "validation_prompt": MOCK_VALIDATION_PROMPT}):
        return TextSummarizerTool(mock_llm_client)

def test__run(text_summarizer_tool):
    output = text_summarizer_tool._run(INPUT_TEXT)
    assert output["summary"] == EXPECTED_SUMMARY
    assert output["factualness_score"] == EXPECTED_FACTUALNESS_SCORE
    assert output["completeness_score"] == EXPECTED_COMPLETENESS_SCORE

def test__arun(text_summarizer_tool):
    output = text_summarizer_tool._arun(INPUT_TEXT)
    assert output["summary"] == EXPECTED_SUMMARY
    assert output["factualness_score"] == EXPECTED_FACTUALNESS_SCORE
    assert output["completeness_score"] == EXPECTED_COMPLETENESS_SCORE

def test__arun_async(text_summarizer_tool):
    with pytest.raises(NotImplementedError):
        text_summarizer_tool._arun_async(INPUT_TEXT)

def test_low_factualness_score(text_summarizer_tool, monkeypatch):
    monkeypatch.setattr(SummaryValidationOutput, "parse_obj", lambda _: SummaryValidationOutput(factualness_score=0.7, completeness_score=0.8))
    output = text_summarizer_tool._run(INPUT_TEXT)
    assert output["summary"] == EXPECTED_SUMMARY
    assert output["factualness_score"] == EXPECTED_FACTUALNESS_SCORE
    assert output["completeness_score"] == EXPECTED_COMPLETENESS_SCORE
