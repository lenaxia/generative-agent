from langchain_core.tools.base import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PythonLoader
from langchain_chroma import Chroma
import os
from langchain.document_loaders import PythonLoader, TextLoader
from langchain_text_splitters.python import PythonCodeTextSplitter
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

from agents.coding_agent.types import CodingState, WriteCode


from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

class use_aider(BaseTool):
    name: str = "use_aider"
    description: str = "Use this tool when needing to make code edits. Input should be a problem/prompt description, and a list of files to edit, and an optional read only list for context"
    def _run(self, state: CodingState) -> CodingState:
        io = InputOutput(yes=True)
        model = Model("bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0")
        
        coder = Coder.create(main_model=model, fnames=state["context"], io=io)
        
        coder.run(state["prompt"])
        
        return state
        


class draft_code(BaseTool):
    name: str = "draft_code"
    description: str = "Useful for when you need to draft code. Input should be a problem description and code to modify. Output should be a draft code solution in the language defined in the prompt."
    
    def _run(self, state: CodingState) -> CodingState:
        draft_parser = PydanticOutputParser(pydantic_object=WriteCode)

        draft_prompt = PromptTemplate(
            input_variables=["prompt", "context", "language"],
            template="Relevant Context:\n```\n{context}\n```\n\n"
                    "Given the following programming problem description and code to modify, provide a draft {language} code solution. Refernce the code above when necessary.\n\n"
                    "Prompt: {prompt}\n\n"
                    "Your output should follow the provided JSON schema:\n\n{format_instructions}\n\n"
                    "Only respond with the JSON and no additional formatting or comments, do not include anything else besides the json output\n"
                    "Draft Solution:\n",
            partial_variables={"format_instructions": draft_parser.get_format_instructions()},
        )

        draft_chain = draft_prompt | state["llm"] | draft_parser

        draft = draft_chain.invoke({"prompt": state["prompt"], "context": state["context"], "language": state["language"]})
        
        state["candidate"] = draft
        state["messages"].append(draft)
        state["status"] = "draft"
        
        return state
        
        
class retrieve_context(BaseTool):
    name: str = "retrieve_context"
    description: str = "Useful for when you need to retrieve relevant context. Input should be a problem description, code to modify and a corpus of docs such as a code repo. Output should be the relevant context from the corpus."
    
    def _run(self, state: CodingState) -> CodingState:
        docs = self.load_files("/root/personal/generative-agent-test2/supervisor")
        
        text_splitter = PythonCodeTextSplitter(chunk_size=2000, chunk_overlap=500)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=BedrockEmbeddings())
        
        vector_retriever = vectorstore.as_retriever()
        vector_results = vector_retriever.get_relevant_documents(state["prompt"])
        
        #bm25_retriever = BM25Retriever.from_documents(splits)
        #semantic_results = bm25_retriever.get_relevant_documents(state["prompt"])
        
        docs = [doc.metadata['source'] for doc in vector_results]
        
        state["context"] = docs
               
        return state

    def load_files(self, base_dir):
        documents = []

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    python_loader = PythonLoader(file_path)
                    documents.extend(python_loader.load())
                elif file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    markdown_loader = TextLoader(file_path)
                    documents.extend(markdown_loader.load())

        return documents