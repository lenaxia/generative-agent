"""Phase 1 Validation Script

Validates that all Phase 1 components can be imported and work together.
Tests basic functionality without requiring full system initialization.
"""

import sys
from pathlib import Path

def validate_imports():
    """Test that all Phase 1 components can be imported."""
    print("=" * 60)
    print("Phase 1 Validation: Testing Imports")
    print("=" * 60)

    try:
        # Test ToolRegistry import
        from llm_provider.tool_registry import ToolRegistry
        print("✓ ToolRegistry imported successfully")

        # Test IntentCollector imports
        from common.intent_collector import (
            IntentCollector,
            set_current_collector,
            get_current_collector,
            clear_current_collector,
            register_intent,
        )
        print("✓ IntentCollector imported successfully")

        # Test AgentConfiguration import
        from common.agent_configuration import AgentConfiguration
        print("✓ AgentConfiguration imported successfully")

        # Test RuntimeAgentFactory import
        from llm_provider.runtime_agent_factory import RuntimeAgentFactory
        print("✓ RuntimeAgentFactory imported successfully")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def validate_basic_functionality():
    """Test basic functionality of Phase 1 components."""
    print("\n" + "=" * 60)
    print("Phase 1 Validation: Testing Basic Functionality")
    print("=" * 60)

    try:
        # Test 1: IntentCollector
        from common.intent_collector import IntentCollector, set_current_collector, get_current_collector
        from common.intents import Intent

        collector = IntentCollector()
        print(f"✓ Created IntentCollector (count: {collector.count()})")

        # Create a mock intent for testing
        class TestIntent(Intent):
            def __init__(self):
                super().__init__()
                self.intent_type = "test"

            def validate(self) -> bool:
                """Validate the test intent."""
                return True

        test_intent = TestIntent()
        collector.register(test_intent)
        print(f"✓ Registered test intent (count: {collector.count()})")

        intents = collector.get_intents()
        assert len(intents) == 1, "Should have 1 intent"
        print(f"✓ Retrieved intents (count: {len(intents)})")

        # Test context-local storage
        set_current_collector(collector)
        retrieved = get_current_collector()
        assert retrieved is collector, "Should retrieve same collector"
        print("✓ Context-local storage working")

        # Test 2: AgentConfiguration
        from common.agent_configuration import AgentConfiguration

        config = AgentConfiguration(
            plan="Step 1: Do something\nStep 2: Do something else",
            system_prompt="You are a test agent.",
            tool_names=["weather.get_forecast", "calendar.list_events"],
            guidance="Be careful and thorough.",
            max_iterations=10,
            metadata={"test": True},
        )
        print("✓ Created AgentConfiguration")

        # Test validation
        is_valid = config.validate()
        assert is_valid, "Configuration should be valid"
        print(f"✓ AgentConfiguration validation passed")

        # Test serialization
        config_dict = config.to_dict()
        assert "plan" in config_dict, "Should have plan in dict"
        print(f"✓ AgentConfiguration serialization working")

        # Test deserialization
        config2 = AgentConfiguration.from_dict(config_dict)
        assert config2.plan == config.plan, "Should deserialize correctly"
        print(f"✓ AgentConfiguration deserialization working")

        # Test 3: ToolRegistry
        from llm_provider.tool_registry import ToolRegistry

        registry = ToolRegistry()
        print(f"✓ Created ToolRegistry (loaded: {registry.is_loaded()})")

        # Test summary
        summary = registry.get_tool_summary()
        print(f"✓ Got registry summary: {summary['total_tools']} tools, {summary['total_categories']} categories")

        # Test 4: RuntimeAgentFactory (creation only, no actual agent creation)
        from llm_provider.runtime_agent_factory import RuntimeAgentFactory
        from llm_provider.factory import LLMFactory

        # Note: We can't fully test RuntimeAgentFactory without LLMFactory initialization
        # Just test that it can be instantiated
        # factory = RuntimeAgentFactory(registry, llm_factory)
        print("✓ RuntimeAgentFactory class available (full test requires LLMFactory)")

        return True

    except Exception as e:
        import traceback
        print(f"✗ Functionality test failed: {e}")
        print(traceback.format_exc())
        return False


def validate_domain_structure():
    """Validate domain directory structure."""
    print("\n" + "=" * 60)
    print("Phase 1 Validation: Testing Domain Structure")
    print("=" * 60)

    expected_domains = [
        "weather",
        "calendar",
        "timer",
        "smart_home",
        "memory",
        "search",
        "notification",
        "planning",
    ]

    roles_dir = Path("roles")

    if not roles_dir.exists():
        print(f"✗ roles/ directory not found")
        return False

    all_good = True
    for domain in expected_domains:
        domain_dir = roles_dir / domain
        if domain_dir.exists():
            # Check for required files
            init_file = domain_dir / "__init__.py"
            tools_file = domain_dir / "tools.py"

            if init_file.exists():
                print(f"✓ {domain}/__init__.py exists")
            else:
                print(f"✗ {domain}/__init__.py missing")
                all_good = False

            if tools_file.exists():
                print(f"✓ {domain}/tools.py exists")
            else:
                print(f"✗ {domain}/tools.py missing")
                all_good = False
        else:
            print(f"✗ {domain}/ directory missing")
            all_good = False

    return all_good


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 COMPONENT VALIDATION")
    print("=" * 60 + "\n")

    results = []

    # Test imports
    results.append(("Imports", validate_imports()))

    # Test basic functionality
    results.append(("Basic Functionality", validate_basic_functionality()))

    # Test domain structure
    results.append(("Domain Structure", validate_domain_structure()))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All Phase 1 validation tests PASSED!")
        return 0
    else:
        print("\n✗ Some Phase 1 validation tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
