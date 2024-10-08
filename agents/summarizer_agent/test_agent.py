import unittest
from unittest.mock import Mock, patch
from agents.summarizer_agent.agent import TextSummarizerAgent, TextSummarizeInput, TextSummarizeOutput
from llm_provider.factory import LLMFactory
from shared_tools.message_bus import MessageBus
from logging import Logger

class TestTextSummarizerAgent(unittest.TestCase):
    def setUp(self):
        self.logger = Mock(Logger)
        self.llm_factory = Mock(LLMFactory)
        self.message_bus = Mock(MessageBus)
        self.agent_id = "text_summarizer_agent"
        self.config = {
            "chunk_size": 500,
            "max_summary_length": 100,
            "factualness_threshold": 0.8,
            "completeness_threshold": 0.8,
            "relevance_threshold": 0.8
        }
        self.agent = TextSummarizerAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)

    @patch("agents.summarizer_agent.agent.TextSummarizerAgent._select_llm_provider")
    @patch("agents.summarizer_agent.agent.TextSummarizerAgent.calculate_accuracy_scores")
    @patch("agents.summarizer_agent.agent.TextSummarizerAgent.calculate_relevance_score")
    def test_run(self, mock_relevance_score, mock_accuracy_scores, mock_select_llm_provider):
        mock_llm_provider = Mock()
        mock_select_llm_provider.return_value = mock_llm_provider
        mock_accuracy_scores.return_value = (0.9, 0.9)
        mock_relevance_score.return_value = 0.9
        input_data = TextSummarizeInput(text="This is a sample text to summarize.", max_summary_length=100)
        expected_output = TextSummarizeOutput(summary="Summary text", factualness_score=0.9, completeness_score=0.9, relevance_score=0.9)
        mock_llm_provider.arun.return_value = {"summary": expected_output.summary}
        

        output = self.agent._run(mock_llm_provider, input_data)

        self.assertEqual(output, expected_output)
        mock_select_llm_provider.assert_called_once()
        mock_accuracy_scores.assert_called_once_with(expected_output.summary, input_data.text)
        mock_relevance_score.assert_called_once_with(expected_output.summary, input_data.text)

    def test_format_input(self):
        instruction = "Summarize this text"
        text = "This is a sample text to summarize."
        max_summary_length = 100
        expected_input = TextSummarizeInput(text=text, max_summary_length=max_summary_length)

        input_data = self.agent._format_input(instruction, text, max_summary_length)

        self.assertEqual(input_data, expected_input)

    def test_process_output(self):
        output_data = TextSummarizeOutput(summary="Summary text", factualness_score=0.9, completeness_score=0.9, relevance_score=0.9)
        expected_output = f"Summary: {output_data.summary}\nFactualness Score: {output_data.factualness_score}\nCompleteness Score: {output_data.completeness_score}\nRelevance Score: {output_data.relevance_score}"

        output = self.agent._process_output(output_data)

        self.assertEqual(output, expected_output)

    def test_calculate_accuracy_scores(self):
        summary = "Summary text"
        original_text = "This is a sample text to summarize."
        expected_factualness_score = 0.8
        expected_completeness_score = 0.7

        factualness_score, completeness_score = self.agent.calculate_accuracy_scores(summary, original_text)

        self.assertEqual(factualness_score, expected_factualness_score)
        self.assertEqual(completeness_score, expected_completeness_score)

    def test_calculate_relevance_score(self):
        summary = "Summary text"
        original_text = "This is a sample text to summarize."
        expected_relevance_score = 0.5  # Assuming a dummy value for the test

        with patch("agents.summarizer_agent.agent.cosine_similarity") as mock_cosine_similarity:
            mock_cosine_similarity.return_value = [[expected_relevance_score]]
            relevance_score = self.agent.calculate_relevance_score(summary, original_text)

        self.assertEqual(relevance_score, expected_relevance_score)

if __name__ == "__main__":
    unittest.main()