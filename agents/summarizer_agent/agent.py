from logging import Logger
from langchain.tools import BaseTool
from shared_tools.message_bus import MessageBus
from agents.base_agent import BaseAgent

from typing import List, Literal, TypedDict, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph


from typing import Any, Dict, List
from pydantic import BaseModel
from llm_provider.factory import LLMFactory, LLMType
from logging import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextSummarizeInput(BaseModel):
    text: str
    max_summary_length: int

class TextSummarizeOutput(BaseModel):
    summary: str
    factualness_score: float
    completeness_score: float
    relevance_score: float

class SummarizationState(TypedDict):
    contents: List[str]
    index: int
    summary: str

class TextSummarizerAgent(BaseAgent):
    def __init__(self, logger: Logger, llm_factory: LLMFactory, message_bus: MessageBus, agent_id: str, config: Dict = None):
        super().__init__(logger, llm_factory, message_bus, agent_id, config)
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 500)
        self.max_summary_length = self.config.get("max_summary_length", 500)
        self.factualness_threshold = self.config.get("factualness_threshold", 0.8)
        self.completeness_threshold = self.config.get("completeness_threshold", 0.8)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.8)
        self.setup_graph()

    @property
    def tools(self) -> Dict[str, BaseTool]:
        # No tools needed for this agent
        return {}

    def _select_llm_provider(self, llm_type: LLMType, **kwargs):
        llm = self.llm_factory.create_chat_model(llm_type)
        return llm

    def _run(self, llm_provider, input_data: TextSummarizeInput) -> TextSummarizeOutput:
        # Split the input text into chunks
        #chunks = [input_data.text[i:i+self.chunk_size] for i in range(0, len(input_data.text), self.chunk_size)]

        # Run the iterative summarization process
        #config = {"configurable": {"thread_id": "abc123"}}
        #output = self.app.invoke({"contents": chunks}, config=config)
        #print(output)
        #print(type(output))
        

        # Calculate factualness, completeness, and relevance scores
        #factualness_score, completeness_score = self.calculate_accuracy_scores(summary, input_data.text)
        #relevance_score = self.calculate_relevance_score(summary, input_data.text)

        # Check if scores meet thresholds, regenerate summary if needed
        #if factualness_score < self.factualness_threshold or completeness_score < self.completeness_threshold or relevance_score < self.relevance_threshold:
        #    self.logger.info("Summary scores did not meet thresholds, regenerating summary.")
        #    return self._run(llm_provider, input_data)

        #return TextSummarizeOutput(summary=summary, factualness_score=factualness_score, completeness_score=completeness_score, relevance_score=relevance_score)

        return TextSummarizeOutput(summary="hello world", factualness_score=0.9, completeness_score=0.9, relevance_score=0.9)

    def _arun(self, llm_provider, input_data: TextSummarizeInput) -> TextSummarizeOutput:
        raise NotImplementedError("Asynchronous execution not supported.")

    def _format_input(self, instruction: str, text: str, max_summary_length: int) -> TextSummarizeInput:
        return TextSummarizeInput(text=text, max_summary_length=max_summary_length)

    def _process_output(self, output: TextSummarizeOutput) -> str:
        return f"Summary: {output.summary}\nFactualness Score: {output.factualness_score}\nCompleteness Score: {output.completeness_score}\nRelevance Score: {output.relevance_score}"

    def setup(self):
        pass

    def teardown(self):
        pass

    def setup_graph(self):
        # Initial summary
        summarize_prompt = ChatPromptTemplate(
            [
                ("human", f"Write a concise summary of the following in {self.max_summary_length} words or less: {{context}}"),
            ]
        )
        initial_summary_chain = summarize_prompt | self._select_llm_provider(LLMType.DEFAULT) | StrOutputParser()

        # Refining the summary with new docs
        refine_template = """
        Produce a final summary in {max_summary_length} words or less.

        Existing summary up to this point:
        {existing_answer}

        New context:
        ------------
        {context}
        ------------

        Given the new context, refine the original summary.
        """
        refine_prompt = ChatPromptTemplate([("human", refine_template)])

        refine_summary_chain = refine_prompt | self._select_llm_provider(LLMType.DEFAULT) | StrOutputParser()

        def generate_initial_summary(state: SummarizationState, config: RunnableConfig):
            summary = initial_summary_chain.invoke(
                state["contents"][0],
                config,
            )
            return {"summary": summary, "index": 1}


        def refine_summary(state: SummarizationState, config: RunnableConfig):
            content = state["contents"][state["index"]]
            summary = refine_summary_chain.invoke(
                {"existing_answer": state["summary"], "context": content, "max_summary_length": self.max_summary_length},
                config,
            )

            return{"summary": summary, "index": state["index"] + 1}

        def should_refine(state: SummarizationState) -> str:
            if state["index"] >= len(state["contents"]):
                return END
            else:
                return "refine_summary"

        graph = StateGraph(SummarizationState)
        graph.add_node("generate_initial_summary", generate_initial_summary)
        graph.add_node("refine_summary", refine_summary)

        graph.add_edge(START, "generate_initial_summary")
        graph.add_conditional_edges("generate_initial_summary", should_refine)
        graph.add_conditional_edges("refine_summary", should_refine)
        self.app = graph.compile()

    def calculate_accuracy_scores(self, summary, original_text):
        # TODO: Implement factualness and completeness scoring logic here
        # For now, returning dummy scores
        factualness_score = 0.8
        completeness_score = 0.7
        return factualness_score, completeness_score

    def calculate_relevance_score(self, summary, original_text):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([summary, original_text])
        relevance_score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return relevance_score

# Example configuration dictionary
#example_config = {
#    "chunk_size": 1000,  # Adjust chunk size for splitting input text
#    "max_summary_length": 200,  # Maximum length of the summary
#    "factualness_threshold": 0.9,  # Threshold for factualness score
#    "completeness_threshold": 0.9,  # Threshold for completeness score
#    "relevance_threshold": 0.9,  # Threshold for relevance score
#}