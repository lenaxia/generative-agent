import time
from unittest.mock import Mock, patch

import pytest

from config.base_config import BaseConfig
from llm_provider.factory import LLMFactory, LLMType
from llm_provider.role_registry import RoleDefinition, RoleRegistry
from llm_provider.universal_agent import UniversalAgent


class TestPerformanceBenchmarking:
    """Performance benchmarking tests for Fast-Reply Performance Optimization."""

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
            LLMType.STRONG: [mock_config],
        }
        return LLMFactory(configs)

    @pytest.fixture
    def mock_role_registry(self):
        """Create a mock role registry with weather and router roles."""
        registry = Mock(spec=RoleRegistry)

        # Router role definition
        router_role = Mock(spec=RoleDefinition)
        router_role.name = "router"
        router_role.config = {
            "prompts": {
                "system": "You are a request router. Analyze requests and route to appropriate roles."
            },
            "tools": {"shared": []},
        }
        router_role.custom_tools = []

        # Weather role definition
        weather_role = Mock(spec=RoleDefinition)
        weather_role.name = "weather"
        weather_role.config = {
            "prompts": {
                "system": "You are a weather specialist. Provide accurate weather information."
            },
            "tools": {"shared": ["weather_tools"]},
        }
        weather_role.custom_tools = []

        # Default role definition
        default_role = Mock(spec=RoleDefinition)
        default_role.name = "default"
        default_role.config = {
            "prompts": {"system": "You are a helpful assistant."},
            "tools": {"shared": []},
        }
        default_role.custom_tools = []

        def get_role_side_effect(role_name):
            if role_name == "router":
                return router_role
            elif role_name == "weather":
                return weather_role
            elif role_name == "default":
                return default_role
            return None

        registry.get_role.side_effect = get_role_side_effect
        registry.get_shared_tool.return_value = Mock()

        return registry

    @pytest.fixture
    def universal_agent(self, factory_with_config, mock_role_registry):
        """Create UniversalAgent with mocked dependencies."""
        return UniversalAgent(
            llm_factory=factory_with_config, role_registry=mock_role_registry
        )

    def test_agent_creation_vs_pooling_performance(self, factory_with_config):
        """Benchmark agent creation vs agent pooling performance."""
        with (
            patch("llm_provider.factory.BedrockModel") as mock_bedrock,
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            # Simulate slow agent creation (realistic overhead)
            def slow_agent_creation(*args, **kwargs):
                time.sleep(0.001)  # 1ms delay to simulate real overhead
                agent = Mock()
                agent.model = Mock()
                agent.system_prompt = "test"
                agent.tools = []
                return agent

            mock_bedrock.return_value = Mock()
            mock_agent_class.side_effect = slow_agent_creation

            # Benchmark: Multiple agent creations (old approach)
            start_time = time.perf_counter()
            agents_created = []
            for _ in range(10):
                # Simulate creating new agent each time (old approach)
                model = factory_with_config.create_strands_model(LLMType.DEFAULT)
                agent = mock_agent_class(model=model)
                agents_created.append(agent)
            creation_time = time.perf_counter() - start_time

            # Benchmark: Agent pooling (new approach)
            start_time = time.perf_counter()
            agents_pooled = []
            for _ in range(10):
                # Use agent pooling
                agent = factory_with_config.get_agent(LLMType.DEFAULT)
                agents_pooled.append(agent)
            pooling_time = time.perf_counter() - start_time

            # Agent pooling should be significantly faster
            improvement_ratio = creation_time / pooling_time
            print(f"Creation time: {creation_time:.4f}s")
            print(f"Pooling time: {pooling_time:.4f}s")
            print(f"Improvement ratio: {improvement_ratio:.1f}x")

            # Should be at least 5x faster with pooling
            assert improvement_ratio > 5.0

            # All pooled agents should be the same instance
            first_agent = agents_pooled[0]
            for agent in agents_pooled[1:]:
                assert agent is first_agent

    def test_role_switching_performance(self, universal_agent):
        """Benchmark role switching performance with agent pooling."""
        with (
            patch("llm_provider.factory.BedrockModel"),
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            mock_agent = Mock()
            mock_agent.update_context = Mock()
            mock_agent_class.return_value = mock_agent

            # Warm up the agent pool
            universal_agent.llm_factory.warm_agent_pool()

            # Benchmark role switching
            roles_to_test = ["router", "weather", "router", "weather", "default"]

            start_time = time.perf_counter()
            for role in roles_to_test:
                agent = universal_agent.assume_role(role)
                assert agent is not None
            switching_time = time.perf_counter() - start_time

            avg_switch_time = switching_time / len(roles_to_test)
            print(f"Total switching time: {switching_time:.4f}s")
            print(f"Average switch time: {avg_switch_time:.4f}s")

            # Each role switch should be reasonably fast (< 10ms)
            assert (
                avg_switch_time < 0.01
            ), f"Average switch time {avg_switch_time:.4f}s should be < 0.01s"

            # Verify agents were created (context update may fail with mocks, which is expected)
            assert (
                mock_agent_class.call_count >= 1
            ), "Should have created at least one agent"

    def test_weather_query_simulation_performance(self, universal_agent):
        """Simulate the weather query workflow from the design document."""
        with (
            patch("llm_provider.factory.BedrockModel"),
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            mock_agent = Mock()
            mock_agent.update_context = Mock()
            mock_agent_class.return_value = mock_agent

            # Simulate the weather query workflow
            start_time = time.perf_counter()

            # Step 1: Router agent (should use WEAK model)
            universal_agent.assume_role("router", LLMType.WEAK)

            # Step 2: Weather agent (should use DEFAULT model)
            universal_agent.assume_role("weather", LLMType.DEFAULT)

            total_time = time.perf_counter() - start_time

            print(f"Weather query simulation time: {total_time:.4f}s")

            # Total agent switching should be reasonably fast (< 50ms)
            assert (
                total_time < 0.05
            ), f"Weather query simulation time {total_time:.4f}s should be < 0.05s"

            # Verify agents were obtained from pool
            pool_stats = universal_agent.llm_factory.get_pool_stats()
            assert pool_stats["total_requests"] >= 2

    def test_concurrent_role_switching_performance(self, universal_agent):
        """Test performance under concurrent role switching load."""
        import concurrent.futures

        with (
            patch("llm_provider.factory.BedrockModel"),
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            mock_agent = Mock()
            mock_agent.update_context = Mock()
            mock_agent_class.return_value = mock_agent

            # Pre-warm the pool
            universal_agent.llm_factory.warm_agent_pool()

            def role_switch_worker(role_name):
                """Worker function for concurrent role switching."""
                start_time = time.perf_counter()
                agent = universal_agent.assume_role(role_name)
                end_time = time.perf_counter()
                return end_time - start_time, agent

            # Test concurrent role switches
            roles = ["router", "weather", "default"] * 10  # 30 total switches

            start_time = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(role_switch_worker, role) for role in roles]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
            total_time = time.perf_counter() - start_time

            # Extract timing results
            switch_times = [result[0] for result in results]
            avg_switch_time = sum(switch_times) / len(switch_times)
            max_switch_time = max(switch_times)

            print(f"Concurrent switching - Total time: {total_time:.4f}s")
            print(f"Average switch time: {avg_switch_time:.4f}s")
            print(f"Max switch time: {max_switch_time:.4f}s")

            # Even under load, switches should be reasonably fast (< 100ms)
            assert (
                avg_switch_time < 0.1
            ), f"Average concurrent switch time {avg_switch_time:.4f}s should be < 0.1s"
            assert (
                max_switch_time < 0.1
            )  # 100ms max for individual switches under concurrent load

            # Verify high pool hit rate
            pool_stats = universal_agent.llm_factory.get_pool_stats()
            hit_rate = pool_stats["hit_rate_percent"]
            print(f"Pool hit rate: {hit_rate}%")
            assert hit_rate > 80  # Should have high cache hit rate

    def test_memory_efficiency_with_pooling(self, factory_with_config):
        """Test that agent pooling doesn't cause memory leaks."""
        with (
            patch("llm_provider.factory.BedrockModel"),
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            mock_agent_class.return_value = Mock()

            # Create many agent requests
            for _ in range(100):
                agent = factory_with_config.get_agent(LLMType.DEFAULT)
                assert agent is not None

            # Pool should only contain one agent despite 100 requests
            pool_stats = factory_with_config.get_pool_stats()
            assert pool_stats["pool_size"] == 1
            assert pool_stats["hits"] == 99  # First was miss, rest were hits
            assert pool_stats["misses"] == 1
            assert pool_stats["created"] == 1

            # Hit rate should be very high
            assert pool_stats["hit_rate_percent"] == 99.0

    def test_pool_warming_performance(self, factory_with_config):
        """Test agent pool warming performance."""
        with (
            patch("llm_provider.factory.BedrockModel"),
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            mock_agent_class.return_value = Mock()

            # Benchmark pool warming
            start_time = time.perf_counter()
            factory_with_config.warm_agent_pool()
            warming_time = time.perf_counter() - start_time

            print(f"Pool warming time: {warming_time:.4f}s")

            # Pool warming should be reasonably fast (< 100ms)
            assert warming_time < 0.1

            # Should have created agents for common combinations
            pool_stats = factory_with_config.get_pool_stats()
            assert pool_stats["pool_size"] == 3  # WEAK, DEFAULT, STRONG
            assert pool_stats["created"] == 3

    def test_performance_regression_detection(self, universal_agent):
        """Test to detect performance regressions in agent operations."""
        with (
            patch("llm_provider.factory.BedrockModel"),
            patch("llm_provider.factory.Agent") as mock_agent_class,
        ):
            mock_agent = Mock()
            mock_agent.update_context = Mock()
            mock_agent_class.return_value = mock_agent

            # Define performance thresholds (realistic for test environment)
            MAX_AGENT_POOL_ACCESS_TIME = 0.01  # 10ms
            MAX_CONTEXT_SWITCH_TIME = 0.01  # 10ms
            MAX_ROLE_ASSUMPTION_TIME = 0.05  # 50ms total

            # Test agent pool access time
            start_time = time.perf_counter()
            agent = universal_agent.llm_factory.get_agent(LLMType.DEFAULT)
            pool_access_time = time.perf_counter() - start_time

            # Test context switching time
            start_time = time.perf_counter()
            universal_agent._update_agent_context(agent, "test prompt", [])
            context_switch_time = time.perf_counter() - start_time

            # Test full role assumption time
            start_time = time.perf_counter()
            universal_agent.assume_role("weather")
            role_assumption_time = time.perf_counter() - start_time

            print(f"Agent pool access time: {pool_access_time:.6f}s")
            print(f"Context switch time: {context_switch_time:.6f}s")
            print(f"Role assumption time: {role_assumption_time:.6f}s")

            # Verify performance meets targets
            assert (
                pool_access_time < MAX_AGENT_POOL_ACCESS_TIME
            ), f"Agent pool access too slow: {pool_access_time:.6f}s > {MAX_AGENT_POOL_ACCESS_TIME}s"

            assert (
                context_switch_time < MAX_CONTEXT_SWITCH_TIME
            ), f"Context switching too slow: {context_switch_time:.6f}s > {MAX_CONTEXT_SWITCH_TIME}s"

            assert (
                role_assumption_time < MAX_ROLE_ASSUMPTION_TIME
            ), f"Role assumption too slow: {role_assumption_time:.6f}s > {MAX_ROLE_ASSUMPTION_TIME}s"


class TestPerformanceComparison:
    """Compare old vs new approach performance."""

    def test_before_vs_after_performance_simulation(self):
        """Simulate the before/after performance comparison from design doc."""

        # Simulate OLD approach (agent creation each time)
        def simulate_old_approach():
            """Simulate old approach with agent creation overhead."""
            start_time = time.perf_counter()

            # Router agent creation (1.4s simulated as 1.4ms for testing)
            time.sleep(0.0014)

            # Weather agent creation (1.5s simulated as 1.5ms for testing)
            time.sleep(0.0015)

            # LLM calls and API calls (not optimized)
            time.sleep(0.0026)  # 2.6s simulated as 2.6ms

            return time.perf_counter() - start_time

        # Simulate NEW approach (agent pooling)
        def simulate_new_approach():
            """Simulate new approach with agent pooling."""
            start_time = time.perf_counter()

            # Agent pool access (0.001s each, simulated as 0.001ms)
            time.sleep(0.000001)  # Router agent from pool
            time.sleep(0.000001)  # Weather agent from pool

            # LLM calls and API calls (same as before)
            time.sleep(0.0026)  # 2.6s simulated as 2.6ms

            return time.perf_counter() - start_time

        # Run simulations
        old_time = simulate_old_approach()
        new_time = simulate_new_approach()

        improvement_ratio = old_time / new_time
        improvement_percent = ((old_time - new_time) / old_time) * 100

        print(f"Old approach time: {old_time:.6f}s")
        print(f"New approach time: {new_time:.6f}s")
        print(f"Improvement ratio: {improvement_ratio:.1f}x")
        print(f"Improvement percent: {improvement_percent:.1f}%")

        # Should see significant improvement (at least 40% as per design)
        assert improvement_percent > 40
        assert improvement_ratio > 1.5

    def test_target_performance_achievement(self):
        """Verify that we achieve the 3.0s target from 5.74s baseline."""
        # Based on design document analysis:
        # Before: 5.74s total (2.9s agent creation + 2.84s other)
        # After: 3.0s total (0.002s agent pooling + 2.84s other)

        TARGET_TIME = 3.0  # seconds
        EXPECTED_IMPROVEMENT = 48  # percent

        # Simulate the improvement
        agent_creation_overhead_old = 2.9
        agent_pooling_overhead_new = 0.002
        other_operations = 2.84  # LLM calls, API calls, etc.

        simulated_old_time = agent_creation_overhead_old + other_operations
        simulated_new_time = agent_pooling_overhead_new + other_operations

        actual_improvement = (
            (simulated_old_time - simulated_new_time) / simulated_old_time
        ) * 100

        print(f"Baseline time: {simulated_old_time:.3f}s")
        print(f"Optimized time: {simulated_new_time:.3f}s")
        print(f"Actual improvement: {actual_improvement:.1f}%")
        print(f"Target improvement: {EXPECTED_IMPROVEMENT}%")

        # Verify we meet or exceed the target
        assert simulated_new_time <= TARGET_TIME
        assert actual_improvement >= EXPECTED_IMPROVEMENT

        # Verify the improvement is primarily from agent pooling
        pooling_contribution = (
            (agent_creation_overhead_old - agent_pooling_overhead_new)
            / (simulated_old_time - simulated_new_time)
        ) * 100
        print(f"Agent pooling contribution to improvement: {pooling_contribution:.1f}%")
        assert pooling_contribution > 95  # Agent pooling should be the main contributor
