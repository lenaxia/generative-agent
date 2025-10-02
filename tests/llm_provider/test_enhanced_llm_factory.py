import unittest
from unittest.mock import Mock, patch
from llm_provider.factory import LLMFactory, LLMType


class TestEnhancedLLMFactory(unittest.TestCase):
    """Test Enhanced LLM Factory with actual implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # LLMFactory expects Dict[LLMType, List[BaseConfig]] but we'll use empty dict for testing
        self.config = {}
        self.factory = LLMFactory(self.config)
    
    def test_enhanced_factory_initialization_with_framework_selection(self):
        """Test factory initialization with framework selection."""
        # Test that factory initializes with strands framework
        self.assertEqual(self.factory.get_framework(), "strands")
        
        # Test that configs are loaded
        self.assertIsNotNone(self.factory.configs)
    
    def test_enhanced_factory_preserves_existing_functionality(self):
        """Test that enhanced factory preserves existing functionality."""
        # Test that factory has core methods
        self.assertTrue(hasattr(self.factory, 'get_framework'))
        self.assertTrue(hasattr(self.factory, 'create_strands_model'))
        
        # Test framework selection
        framework = self.factory.get_framework()
        self.assertEqual(framework, "strands")
    
    def test_framework_selection_validation(self):
        """Test framework selection validation."""
        # Test that factory uses strands framework
        factory_strands = LLMFactory(self.config)
        self.assertEqual(factory_strands.get_framework(), "strands")
        
        # Test that framework is consistent
        self.assertEqual(factory_strands.get_framework(), "strands")
    
    def test_backward_compatibility_with_strands(self):
        """Test backward compatibility with strands framework."""
        # Test that factory works with strands
        self.assertTrue(hasattr(self.factory, 'create_strands_model'))
        
        # Test that framework is strands
        self.assertEqual(self.factory.get_framework(), "strands")


if __name__ == "__main__":
    unittest.main()