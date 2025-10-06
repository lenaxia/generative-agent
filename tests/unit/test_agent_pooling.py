import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from llm_provider.factory import LLMFactory, LLMType
from config.base_config import BaseConfig


class TestAgentPooling:
    """Test suite for Agent pooling functionality in LLMFactory."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=BaseConfig)
        config.name = "test_bedrock"
        config.provider_type = "bedrock"
        config.model_id = "anthropic.claude-sonnet-4-20250514-v1:0"
        config.temperature = 0.3
        config.additional_params = {}
        return config
    
    @pytest.fixture
    def factory_with_config(self, mock_config):
        """Create LLMFactory with test configuration."""
        configs = {
            LLMType.WEAK: [mock_config],
            LLMType.DEFAULT: [mock_config],
            LLMType.STRONG: [mock_config]
        }
        return LLMFactory(configs)
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock Agent for testing."""
        agent = Mock()
        agent.model = Mock()
        agent.system_prompt = "test prompt"
        agent.tools = []
        return agent
    
    def test_agent_pool_initialization(self, factory_with_config):
        """Test that agent pool is properly initialized."""
        factory = factory_with_config
        
        # Agent pool should be initialized as empty dict
        assert hasattr(factory, '_agent_pool')
        assert isinstance(factory._agent_pool, dict)
        assert len(factory._agent_pool) == 0
        
        # Pool stats should be initialized
        assert hasattr(factory, '_pool_stats')
        assert factory._pool_stats['hits'] == 0
        assert factory._pool_stats['misses'] == 0
        assert factory._pool_stats['created'] == 0
    
    @patch('llm_provider.factory.Agent')
    def test_get_agent_creates_new_agent_on_miss(self, mock_agent_class, factory_with_config):
        """Test that get_agent creates new agent when not in pool."""
        factory = factory_with_config
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock the model creation
        with patch.object(factory, 'create_strands_model') as mock_create_model:
            mock_model = Mock()
            mock_create_model.return_value = mock_model
            
            # First call should create new agent
            agent = factory.get_agent(LLMType.DEFAULT)
            
            # Verify agent was created and cached
            assert agent == mock_agent
            assert factory._pool_stats['misses'] == 1
            assert factory._pool_stats['created'] == 1
            assert factory._pool_stats['hits'] == 0
            
            # Verify agent is in pool
            pool_key = "bedrock_default"
            assert pool_key in factory._agent_pool
            assert factory._agent_pool[pool_key] == mock_agent
    
    @patch('llm_provider.factory.Agent')
    def test_get_agent_returns_cached_agent_on_hit(self, mock_agent_class, factory_with_config):
        """Test that get_agent returns cached agent on subsequent calls."""
        factory = factory_with_config
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        with patch.object(factory, 'create_strands_model') as mock_create_model:
            mock_model = Mock()
            mock_create_model.return_value = mock_model
            
            # First call creates agent
            agent1 = factory.get_agent(LLMType.DEFAULT)
            
            # Second call should return cached agent
            agent2 = factory.get_agent(LLMType.DEFAULT)
            
            # Should be same agent instance
            assert agent1 == agent2
            assert agent1 is agent2
            
            # Stats should show one miss and one hit
            assert factory._pool_stats['misses'] == 1
            assert factory._pool_stats['hits'] == 1
            assert factory._pool_stats['created'] == 1
    
    def test_get_agent_with_different_providers(self, factory_with_config):
        """Test that different providers create separate pool entries."""
        factory = factory_with_config
        
        with patch('llm_provider.factory.Agent') as mock_agent_class, \
             patch.object(factory, 'create_strands_model') as mock_create_model:
            
            mock_agent1 = Mock()
            mock_agent2 = Mock()
            mock_agent_class.side_effect = [mock_agent1, mock_agent2]
            mock_create_model.return_value = Mock()
            
            # Get agents with different providers
            agent1 = factory.get_agent(LLMType.DEFAULT, provider="bedrock")
            agent2 = factory.get_agent(LLMType.DEFAULT, provider="openai")
            
            # Should be different agents
            assert agent1 != agent2
            assert agent1 is not agent2
            
            # Should have separate pool entries
            assert "bedrock_default" in factory._agent_pool
            assert "openai_default" in factory._agent_pool
            assert factory._agent_pool["bedrock_default"] == agent1
            assert factory._agent_pool["openai_default"] == agent2
    
    def test_get_agent_with_different_llm_types(self, factory_with_config):
        """Test that different LLM types create separate pool entries."""
        factory = factory_with_config
        
        with patch('llm_provider.factory.Agent') as mock_agent_class, \
             patch.object(factory, 'create_strands_model') as mock_create_model:
            
            mock_agent1 = Mock()
            mock_agent2 = Mock()
            mock_agent_class.side_effect = [mock_agent1, mock_agent2]
            mock_create_model.return_value = Mock()
            
            # Get agents with different LLM types
            agent1 = factory.get_agent(LLMType.WEAK)
            agent2 = factory.get_agent(LLMType.STRONG)
            
            # Should be different agents
            assert agent1 != agent2
            
            # Should have separate pool entries
            assert "bedrock_weak" in factory._agent_pool
            assert "bedrock_strong" in factory._agent_pool
    
    def test_warm_agent_pool(self, factory_with_config):
        """Test agent pool warming functionality."""
        factory = factory_with_config
        
        with patch('llm_provider.factory.Agent') as mock_agent_class, \
             patch.object(factory, 'create_strands_model') as mock_create_model:
            
            mock_agents = [Mock() for _ in range(3)]
            mock_agent_class.side_effect = mock_agents
            mock_create_model.return_value = Mock()
            
            # Warm the pool
            factory.warm_agent_pool()
            
            # Should have created agents for common combinations
            expected_keys = ["bedrock_weak", "bedrock_default", "bedrock_strong"]
            for key in expected_keys:
                assert key in factory._agent_pool
            
            # Pool stats should reflect warming
            assert factory._pool_stats['created'] == 3
            assert len(factory._agent_pool) == 3
    
    def test_warm_agent_pool_handles_errors_gracefully(self, factory_with_config):
        """Test that pool warming handles errors gracefully."""
        factory = factory_with_config
        
        with patch.object(factory, 'get_agent') as mock_get_agent:
            # Make get_agent fail for one type
            mock_get_agent.side_effect = [Mock(), Exception("Test error"), Mock()]
            
            # Warming should not fail
            factory.warm_agent_pool()
            
            # Should have attempted to warm all types
            assert mock_get_agent.call_count == 3
    
    def test_get_pool_stats(self, factory_with_config):
        """Test pool statistics functionality."""
        factory = factory_with_config
        
        with patch('llm_provider.factory.Agent') as mock_agent_class, \
             patch.object(factory, 'create_strands_model') as mock_create_model:
            
            mock_agent_class.return_value = Mock()
            mock_create_model.return_value = Mock()
            
            # Create some pool activity
            factory.get_agent(LLMType.DEFAULT)  # miss + create
            factory.get_agent(LLMType.DEFAULT)  # hit
            factory.get_agent(LLMType.WEAK)     # miss + create
            
            stats = factory.get_pool_stats()
            
            # Verify stats
            assert stats['pool_size'] == 2
            assert stats['hits'] == 1
            assert stats['misses'] == 2
            assert stats['created'] == 2
            assert stats['total_requests'] == 3
            assert stats['hit_rate_percent'] == 33.33  # 1/3 * 100
    
    def test_get_pool_stats_empty_pool(self, factory_with_config):
        """Test pool statistics with empty pool."""
        factory = factory_with_config
        
        stats = factory.get_pool_stats()
        
        assert stats['pool_size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['created'] == 0
        assert stats['total_requests'] == 0
        assert stats['hit_rate_percent'] == 0
    
    def test_get_default_provider(self, factory_with_config):
        """Test _get_default_provider method."""
        factory = factory_with_config
        
        # Should return first available provider
        default_provider = factory._get_default_provider()
        assert default_provider == "bedrock"
    
    def test_agent_pool_thread_safety(self, factory_with_config):
        """Test that agent pool operations are thread-safe."""
        import threading
        import time
        
        factory = factory_with_config
        results = []
        errors = []
        
        def get_agent_worker():
            try:
                with patch('llm_provider.factory.Agent') as mock_agent_class, \
                     patch.object(factory, 'create_strands_model') as mock_create_model:
                    
                    mock_agent_class.return_value = Mock()
                    mock_create_model.return_value = Mock()
                    
                    agent = factory.get_agent(LLMType.DEFAULT)
                    results.append(agent)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_agent_worker)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # All results should be the same agent (from pool)
        assert len(results) == 5
        first_agent = results[0]
        for agent in results[1:]:
            assert agent is first_agent
    
    def test_pool_performance_improvement(self, factory_with_config):
        """Test that agent pooling provides performance improvement."""
        factory = factory_with_config
        
        with patch('llm_provider.factory.Agent') as mock_agent_class, \
             patch.object(factory, 'create_strands_model') as mock_create_model:
            
            # Simulate slow agent creation
            def slow_agent_creation(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                return Mock()
            
            mock_agent_class.side_effect = slow_agent_creation
            mock_create_model.return_value = Mock()
            
            # First call (cache miss) - should be slow
            start_time = time.time()
            agent1 = factory.get_agent(LLMType.DEFAULT)
            first_call_time = time.time() - start_time
            
            # Second call (cache hit) - should be fast
            start_time = time.time()
            agent2 = factory.get_agent(LLMType.DEFAULT)
            second_call_time = time.time() - start_time
            
            # Cache hit should be significantly faster
            assert second_call_time < first_call_time / 2
            assert agent1 is agent2
    
    def test_clear_agent_pool(self, factory_with_config):
        """Test clearing the agent pool."""
        factory = factory_with_config
        
        with patch('llm_provider.factory.Agent') as mock_agent_class, \
             patch.object(factory, 'create_strands_model') as mock_create_model:
            
            mock_agent_class.return_value = Mock()
            mock_create_model.return_value = Mock()
            
            # Create some agents
            factory.get_agent(LLMType.DEFAULT)
            factory.get_agent(LLMType.WEAK)
            
            # Pool should have agents
            assert len(factory._agent_pool) == 2
            
            # Clear the pool
            factory.clear_agent_pool()
            
            # Pool should be empty
            assert len(factory._agent_pool) == 0
            assert factory._pool_stats['hits'] == 0
            assert factory._pool_stats['misses'] == 0
            assert factory._pool_stats['created'] == 0


class TestAgentPoolingIntegration:
    """Integration tests for agent pooling with real components."""
    
    def test_agent_pooling_with_real_config(self):
        """Test agent pooling with realistic configuration."""
        # Create realistic config
        config = Mock(spec=BaseConfig)
        config.name = "bedrock_sonnet"
        config.provider_type = "bedrock"
        config.model_id = "anthropic.claude-sonnet-4-20250514-v1:0"
        config.temperature = 0.3
        config.additional_params = {"region_name": "us-west-2"}
        
        configs = {LLMType.DEFAULT: [config]}
        factory = LLMFactory(configs)
        
        with patch('llm_provider.factory.BedrockModel') as mock_bedrock, \
             patch('llm_provider.factory.Agent') as mock_agent_class:
            
            mock_model = Mock()
            mock_bedrock.return_value = mock_model
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Test multiple calls
            agent1 = factory.get_agent(LLMType.DEFAULT)
            agent2 = factory.get_agent(LLMType.DEFAULT)
            
            # Should be same instance
            assert agent1 is agent2
            
            # Model should only be created once
            mock_bedrock.assert_called_once()
            mock_agent_class.assert_called_once()
    
    def test_agent_pooling_performance_benchmark(self):
        """Benchmark test to verify performance improvement."""
        config = Mock(spec=BaseConfig)
        config.name = "test_config"
        config.provider_type = "bedrock"
        config.model_id = "test-model"
        config.temperature = 0.3
        config.additional_params = {}
        
        configs = {LLMType.DEFAULT: [config]}
        factory = LLMFactory(configs)
        
        with patch('llm_provider.factory.BedrockModel') as mock_bedrock, \
             patch('llm_provider.factory.Agent') as mock_agent_class:
            
            mock_bedrock.return_value = Mock()
            mock_agent_class.return_value = Mock()
            
            # Measure first call (cache miss)
            start_time = time.perf_counter()
            factory.get_agent(LLMType.DEFAULT)
            first_call_duration = time.perf_counter() - start_time
            
            # Measure subsequent calls (cache hits)
            cache_hit_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                factory.get_agent(LLMType.DEFAULT)
                cache_hit_times.append(time.perf_counter() - start_time)
            
            avg_cache_hit_time = sum(cache_hit_times) / len(cache_hit_times)
            
            # Cache hits should be at least 10x faster than first call
            # (accounting for test overhead)
            assert avg_cache_hit_time < first_call_duration / 10
            
            # Verify pool stats
            stats = factory.get_pool_stats()
            assert stats['hits'] == 10
            assert stats['misses'] == 1
            assert stats['hit_rate_percent'] > 90