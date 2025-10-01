import unittest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock langchain dependencies before importing
sys.modules['langchain'] = Mock()
sys.modules['langchain.prompts'] = Mock()
sys.modules['langchain.tools'] = Mock()
sys.modules['langchain_openai'] = Mock()
sys.modules['langchain_aws'] = Mock()
sys.modules['langchain_core'] = Mock()
sys.modules['langchain_core.output_parsers'] = Mock()

from llm_provider.factory import LLMFactory, LLMType
from config.base_config import BaseConfig


class TestEnhancedLLMFactory(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for Enhanced LLMFactory."""
        # Mock configurations for different LLM types
        self.mock_bedrock_config = Mock(spec=BaseConfig)
        self.mock_bedrock_config.name = "bedrock_default"
        self.mock_bedrock_config.provider_name = "bedrock"
        self.mock_bedrock_config.provider_type = "bedrock"
        self.mock_bedrock_config.model_id = "us.amazon.nova-pro-v1:0"
        self.mock_bedrock_config.temperature = 0.3
        self.mock_bedrock_config.additional_params = {}
        self.mock_bedrock_config.llm_config = Mock()
        self.mock_bedrock_config.llm_config.dict = Mock(return_value={
            "model_id": "us.amazon.nova-pro-v1:0",
            "temperature": 0.3
        })
        
        self.mock_openai_config = Mock(spec=BaseConfig)
        self.mock_openai_config.name = "openai_default"
        self.mock_openai_config.provider_name = "openai"
        self.mock_openai_config.provider_type = "openai"
        self.mock_openai_config.model_id = "gpt-4"
        self.mock_openai_config.temperature = 0.3
        self.mock_openai_config.additional_params = {}
        self.mock_openai_config.llm_config = Mock()
        self.mock_openai_config.llm_config.dict = Mock(return_value={
            "model_id": "gpt-4",
            "temperature": 0.3
        })
        
        self.configs = {
            LLMType.DEFAULT: [self.mock_bedrock_config],
            LLMType.STRONG: [self.mock_bedrock_config],
            LLMType.WEAK: [self.mock_openai_config]
        }

    def test_enhanced_factory_initialization_with_framework_selection(self):
        """Test that Enhanced LLMFactory can be initialized with framework selection."""
        # Test with StrandsAgent framework
        factory = LLMFactory(configs=self.configs, framework="strands")
        self.assertEqual(factory.framework, "strands")
        self.assertIsNotNone(factory.prompt_library)
        
        # Test with LangChain framework (backward compatibility)
        factory_langchain = LLMFactory(configs=self.configs, framework="langchain")
        self.assertEqual(factory_langchain.framework, "langchain")

    def test_enhanced_factory_preserves_existing_functionality(self):
        """Test that Enhanced LLMFactory preserves existing LLMFactory functionality."""
        factory = LLMFactory(configs=self.configs)
        
        # Should inherit from LLMFactory
        self.assertIsInstance(factory, LLMFactory)
        
        # Should have access to existing methods
        self.assertTrue(hasattr(factory, 'create_provider'))
        self.assertTrue(hasattr(factory, 'create_chat_model'))
        self.assertTrue(hasattr(factory, 'add_config'))
        self.assertTrue(hasattr(factory, 'remove_config'))

    @patch('llm_provider.factory.BedrockModel')
    def test_create_strands_model_bedrock(self, mock_bedrock_model):
        """Test creating StrandsAgent Bedrock model."""
        factory = LLMFactory(configs=self.configs, framework="strands")
        
        # Mock the BedrockModel
        mock_model_instance = Mock()
        mock_bedrock_model.return_value = mock_model_instance
        
        # Create StrandsAgent model
        model = factory.create_strands_model(LLMType.DEFAULT)
        
        # Verify BedrockModel was called with correct parameters
        mock_bedrock_model.assert_called_once_with(
            model_id="us.amazon.nova-pro-v1:0",
            temperature=0.3
        )
        self.assertEqual(model, mock_model_instance)

    @patch('llm_provider.factory.OpenAIModel')
    def test_create_strands_model_openai(self, mock_openai_model):
        """Test creating StrandsAgent OpenAI model."""
        factory = LLMFactory(configs=self.configs, framework="strands")
        
        # Mock the OpenAIModel
        mock_model_instance = Mock()
        mock_openai_model.return_value = mock_model_instance
        
        # Create StrandsAgent model
        model = factory.create_strands_model(LLMType.WEAK)
        
        # Verify OpenAIModel was called with correct parameters
        mock_openai_model.assert_called_once_with(
            model_id="gpt-4",
            temperature=0.3
        )
        self.assertEqual(model, mock_model_instance)

    def test_create_strands_model_with_named_config(self):
        """Test creating StrandsAgent model with named configuration."""
        
        # Add a named config
        named_config = Mock(spec=BaseConfig)
        named_config.name = "strong_bedrock"
        named_config.provider_type = "bedrock"
        named_config.model_id = "us.amazon.nova-premier-v1:0"
        named_config.temperature = 0.1
        named_config.additional_params = {"max_tokens": 4000}
        
        configs_with_named = self.configs.copy()
        configs_with_named[LLMType.STRONG].append(named_config)
        
        factory = LLMFactory(configs=configs_with_named, framework="strands")
        
        with patch('llm_provider.factory.BedrockModel') as mock_bedrock:
            mock_model = Mock()
            mock_bedrock.return_value = mock_model
            
            # Create model with specific name
            model = factory.create_strands_model(LLMType.STRONG, name="strong_bedrock")
            
            # Verify correct config was used
            mock_bedrock.assert_called_once_with(
                model_id="us.amazon.nova-premier-v1:0",
                temperature=0.1,
                max_tokens=4000
            )

    @patch('llm_provider.factory.Agent')
    def test_create_universal_agent(self, mock_agent_class):
        """Test creating Universal Agent with enhanced factory."""
        factory = LLMFactory(configs=self.configs, framework="strands")
        
        # Mock the Agent class and prompt library
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        mock_prompt = "You are a planning agent. Create comprehensive plans."
        factory.prompt_library.get_prompt = Mock(return_value=mock_prompt)
        
        # Mock tools
        mock_tools = [Mock(), Mock()]
        
        with patch.object(factory, 'create_strands_model') as mock_create_model:
            mock_model = Mock()
            mock_create_model.return_value = mock_model
            
            # Create Universal Agent
            agent = factory.create_universal_agent(
                llm_type=LLMType.STRONG,
                role="planning",
                tools=mock_tools
            )
            
            # Verify model creation
            mock_create_model.assert_called_once_with(LLMType.STRONG)
            
            # Verify prompt retrieval
            factory.prompt_library.get_prompt.assert_called_once_with("planning")
            
            # Verify Agent creation
            mock_agent_class.assert_called_once_with(
                model=mock_model,
                system_prompt=mock_prompt,
                tools=mock_tools
            )
            
            self.assertEqual(agent, mock_agent_instance)

    def test_semantic_model_type_mapping(self):
        """Test that semantic model types are properly mapped."""
        factory = LLMFactory(configs=self.configs, framework="strands")
        
        # Test LLMType enum values are preserved
        self.assertEqual(LLMType.DEFAULT.value, 'default')
        self.assertEqual(LLMType.STRONG.value, 'strong')
        self.assertEqual(LLMType.WEAK.value, 'weak')
        
        # Test factory can access all semantic types
        self.assertIn(LLMType.DEFAULT, factory.configs)
        self.assertIn(LLMType.STRONG, factory.configs)
        self.assertIn(LLMType.WEAK, factory.configs)

    def test_enhanced_factory_error_handling(self):
        """Test error handling in Enhanced LLMFactory."""
        factory = LLMFactory(configs=self.configs, framework="strands")
        
        # Test unsupported provider type
        unsupported_config = Mock(spec=BaseConfig)
        unsupported_config.name = "unsupported"
        unsupported_config.provider_type = "unsupported_provider"
        
        configs_with_unsupported = {LLMType.DEFAULT: [unsupported_config]}
        factory_unsupported = LLMFactory(configs=configs_with_unsupported, framework="strands")
        
        with self.assertRaises(ValueError) as context:
            factory_unsupported.create_strands_model(LLMType.DEFAULT)
        
        self.assertIn("Unsupported provider type", str(context.exception))
        
        # Test missing configuration
        with self.assertRaises(ValueError) as context:
            factory.create_strands_model(LLMType.VISION)  # Not configured
        
        self.assertIn("No configurations found", str(context.exception))

    def test_backward_compatibility_with_langchain(self):
        """Test that enhanced factory maintains backward compatibility with LangChain."""
        factory = LLMFactory(configs=self.configs, framework="langchain")
        
        # Should still be able to use original LangChain methods
        with patch.object(factory, 'create_provider') as mock_create_provider:
            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider
            
            provider = factory.create_provider(LLMType.DEFAULT)
            
            mock_create_provider.assert_called_once_with(LLMType.DEFAULT)
            self.assertEqual(provider, mock_provider)

    def test_prompt_library_integration(self):
        """Test that prompt library is properly integrated."""
        factory = LLMFactory(configs=self.configs, framework="strands")
        
        # Test prompt library exists
        self.assertIsNotNone(factory.prompt_library)
        
        # Test prompt library has expected methods
        self.assertTrue(hasattr(factory.prompt_library, 'get_prompt'))
        self.assertTrue(hasattr(factory.prompt_library, 'add_prompt'))
        self.assertTrue(hasattr(factory.prompt_library, 'list_roles'))

    def test_configuration_management_preserved(self):
        """Test that configuration management from base LLMFactory is preserved."""
        factory = LLMFactory(configs=self.configs, framework="strands")
        
        # Test adding configuration
        new_config = Mock(spec=BaseConfig)
        new_config.name = "new_config"
        
        factory.add_config(LLMType.VISION, new_config)
        self.assertIn(LLMType.VISION, factory.configs)
        self.assertIn(new_config, factory.configs[LLMType.VISION])
        
        # Test removing configuration
        factory.remove_config(LLMType.VISION, "new_config")
        self.assertEqual(len(factory.configs[LLMType.VISION]), 0)

    def test_framework_selection_validation(self):
        """Test framework selection validation."""
        # Valid frameworks should work
        factory_strands = LLMFactory(configs=self.configs, framework="strands")
        self.assertEqual(factory_strands.framework, "strands")
        
        factory_langchain = LLMFactory(configs=self.configs, framework="langchain")
        self.assertEqual(factory_langchain.framework, "langchain")
        
        # Invalid framework should default to strands
        factory_invalid = LLMFactory(configs=self.configs, framework="invalid")
        self.assertEqual(factory_invalid.framework, "strands")


if __name__ == '__main__':
    unittest.main()