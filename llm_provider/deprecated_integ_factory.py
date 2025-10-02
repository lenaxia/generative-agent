import unittest
from langchain.prompts import ChatPromptTemplate
from llm_provider.factory import LLMFactory, LLMType
from config.openai_config import OpenAIConfig

class LLMProviderIntegrationTest(unittest.TestCase):
    def test_llm_provider_invoke(self):
        # Create LLMFactory and populate it with configs
        llm_factory = LLMFactory({})
        openai_config = OpenAIConfig(
            name="openai",
            base_url="http://192.168.5.74:8080/v1/",
            model="qwen2.5-72b-instruct-iq4_xs",
        )
        llm_factory.add_config(LLMType.DEFAULT, openai_config)

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a {mood} assistant, answer the user question"),
            ("user", "What is the meaning of life?")
        ])

        # Create LLM provider
        llm_provider = llm_factory.create_provider(
            llm_type=LLMType.DEFAULT,
            name="openai",
            prompt_template=prompt_template,
        )

        # Make invoke call
        output = llm_provider.invoke({"mood": "sarcastic"})

        # Print the response
        print(str(output))

        # Assert that output is not empty
        self.assertIsNotNone(output)
        self.assertIn("'finish_reason': 'stop'", str(output))


if __name__ == "__main__":
    unittest.main()
