import unittest
from unittest.mock import Mock, patch
from agents.summarizer_agent.agent import TextSummarizerAgent, TextSummarizeInput, TextSummarizeOutput
from llm_provider.factory import LLMFactory, LLMType
from shared_tools.message_bus import MessageBus
from logging import Logger
from langchain_aws import ChatBedrock
from config.bedrock_config import BedrockConfig
from config.openai_config import OpenAIConfig

class TestTextSummarizerAgentIntegration(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(__name__)
        
        # Configure the LLMFactory
        self.llm_factory = LLMFactory({})
        #openai_config = OpenAIConfig(
        #    name="openai",
        #    endpoint="http://192.168.5.74:8080/v1/",
        #    model_name="qwen2.5-32b-instruct",
        #)
        bedrock_config = BedrockConfig(
            name="bedrock", model_id='anthropic.claude-3-sonnet-20240229-v1:0', model_kwargs={'temperature': 0}
        )
        self.llm_factory.add_config(LLMType.DEFAULT, bedrock_config)
        self.message_bus = Mock(MessageBus)
        self.agent_id = "text_summarizer_agent"
        self.config = {
            "chunk_size": 500,
            "max_summary_length": 100,
            "accuracy_threshold": 0.7,
            "completeness_threshold": 0.7,
            "relevance_threshold": 0.7
        }
        self.agent = TextSummarizerAgent(self.logger, self.llm_factory, self.message_bus, self.agent_id, self.config)
        self.agent.setup_graph()


    def test_run_integration(self):
        sample_text = "The Krebs cycle, also known as the citric acid cycle or the tricarboxylic acid (TCA) cycle, is a series of chemical reactions that occur in the matrix of mitochondria in aerobic organisms. It is a crucial part of cellular respiration, which is the process by which cells convert nutrients into energy in the form of ATP (adenosine triphosphate).\n\nThe Krebs cycle serves as a metabolic hub, integrating and processing the breakdown products of carbohydrates, fats, and proteins. It begins with the entry of a two-carbon compound called acetyl-CoA, which is derived from the breakdown of glucose through glycolysis or the breakdown of fatty acids through beta-oxidation. The acetyl-CoA then condenses with a four-carbon molecule called oxaloacetate, forming a six-carbon compound called citrate, which gives the cycle its alternative name.\n\nOver the course of eight successive enzymatic reactions, the citrate molecule is gradually oxidized, releasing two carbon dioxide molecules and generating high-energy electrons that are carried by the electron transport chain to produce ATP through oxidative phosphorylation. The cycle also regenerates oxaloacetate, allowing it to continue in a cyclic fashion.\n\nDuring the Krebs cycle, various intermediates are produced, some of which are used as precursors for other essential biomolecules, such as amino acids, nucleotides, and heme groups. Additionally, the cycle generates NADH and FADH2, which are high-energy electron carriers that feed into the electron transport chain, ultimately driving ATP synthesis.\n\nThe Krebs cycle is a crucial component of aerobic respiration, as it enables the complete oxidation of nutrients and generates a significant portion of the ATP produced during cellular respiration. It is a highly regulated process, with various enzymes and control mechanisms ensuring its proper functioning and integration with other metabolic pathways.\n\n\nThe eight main steps of the Krebs cycle are catalyzed by specific enzymes and involve a series of dehydrogenation, decarboxylation, hydration, and condensation reactions. These steps are as follows:\n\n1. Condensation: Acetyl-CoA combines with oxaloacetate to form citrate, catalyzed by the enzyme citrate synthase.\n\n2. Isomerization: Citrate is isomerized into isocitrate by the enzyme aconitase.\n\n3. Oxidative decarboxylation: Isocitrate is oxidized and decarboxylated to form α-ketoglutarate, releasing CO2 and generating NADH.\n\n4. Oxidative decarboxylation: α-Ketoglutarate is oxidized and decarboxylated to form succinyl-CoA, releasing CO2 and generating NADH.\n\n5. Substrate-level phosphorylation: Succinyl-CoA is converted to succinate, with the energy released being used to phosphorylate GDP to GTP (or ADP to ATP in some organisms).\n\n6. Oxidation: Succinate is oxidized to fumarate, generating FADH2.\n\n7. Hydration: Fumarate is hydrated to form malate, catalyzed by the enzyme fumarase.\n\n8. Oxidation: Malate is oxidized to regenerate oxaloacetate, producing NADH.\n\nAt the end of the cycle, oxaloacetate is regenerated, allowing the cycle to continue. The Krebs cycle is often referred to as an amphibolic pathway because it can operate in both catabolic and anabolic directions, depending on the organism's metabolic needs.\n\nThe Krebs cycle is tightly regulated by various mechanisms, including allosteric regulation of key enzymes, feedback inhibition, and transcriptional control. This regulation ensures that the cycle operates efficiently and in coordination with other metabolic pathways, such as glycolysis, fatty acid oxidation, and the electron transport chain.\n\nFurthermore, the Krebs cycle is connected to various other metabolic pathways through the exchange of intermediates. For example, α-ketoglutarate can be converted to glutamate, an important amino acid, while oxaloacetate can be used in gluconeogenesis to produce glucose.\n\nOverall, the Krebs cycle plays a central role in cellular metabolism, providing a hub for the integration of various catabolic and anabolic pathways and generating a significant portion of the ATP required for cellular processes."

        input_data = TextSummarizeInput(text=sample_text, max_summary_length=100)

        # Create a real LLM provider
        llm_provider = self.llm_factory.create_chat_model(LLMType.DEFAULT)

        output = self.agent._run(llm_provider, input_data)

        self.assertIsInstance(output, TextSummarizeOutput)
        self.assertIsNotNone(output.summary)
        self.assertGreaterEqual(output.accuracy_score, self.config["accuracy_threshold"])
        self.assertGreaterEqual(output.completeness_score, self.config["completeness_threshold"])
        self.assertGreaterEqual(output.relevance_score, self.config["relevance_threshold"])

if __name__ == "__main__":
    unittest.main()