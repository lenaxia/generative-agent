#!/usr/bin/env python3
"""
Hybrid Role Lifecycle Example

This example demonstrates how to use the new hybrid role lifecycle architecture
with the enhanced weather role that includes pre-processing and post-processing.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_provider.role_registry import RoleRegistry
from llm_provider.request_router import RequestRouter
from llm_provider.universal_agent import UniversalAgent
from llm_provider.factory import LLMFactory, LLMType
from supervisor.workflow_engine import WorkflowEngine
from common.message_bus import MessageBus
from common.request_model import RequestMetadata


def demonstrate_hybrid_role_features():
    """Demonstrate the key features of hybrid roles."""
    
    print("🚀 Hybrid Role Lifecycle Architecture Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    role_registry = RoleRegistry("roles")
    
    # Show hybrid role detection
    print(f"   Weather role execution type: {role_registry.get_role_execution_type('weather')}")
    
    # Show parameter schema
    parameters = role_registry.get_role_parameters("weather")
    print(f"   Weather role parameters: {list(parameters.keys())}")
    
    # Show lifecycle functions
    lifecycle_functions = role_registry.get_lifecycle_functions("weather")
    print(f"   Weather lifecycle functions: {list(lifecycle_functions.keys())}")
    
    print("\n2. Parameter Schema Details:")
    for param_name, param_config in parameters.items():
        print(f"   {param_name}:")
        print(f"     - Type: {param_config['type']}")
        print(f"     - Required: {param_config['required']}")
        if 'enum' in param_config:
            print(f"     - Options: {param_config['enum']}")
        if 'default' in param_config:
            print(f"     - Default: {param_config['default']}")
    
    print("\n3. Lifecycle Function Details:")
    role_def = role_registry.get_role("weather")
    
    # Pre-processing functions
    pre_config = role_def.config.get('lifecycle', {}).get('pre_processing', {})
    if pre_config.get('enabled'):
        print("   Pre-processing functions:")
        for func in pre_config.get('functions', []):
            func_name = func.get('name') if isinstance(func, dict) else func
            uses_params = func.get('uses_parameters', []) if isinstance(func, dict) else []
            print(f"     - {func_name}: uses {uses_params}")
    
    # Post-processing functions
    post_config = role_def.config.get('lifecycle', {}).get('post_processing', {})
    if post_config.get('enabled'):
        print("   Post-processing functions:")
        for func in post_config.get('functions', []):
            func_name = func.get('name') if isinstance(func, dict) else func
            print(f"     - {func_name}")


def demonstrate_enhanced_routing():
    """Demonstrate enhanced routing with parameter extraction."""
    
    print("\n🎯 Enhanced Routing with Parameter Extraction")
    print("=" * 50)
    
    # This would normally use real LLM, but for demo we'll show the structure
    print("\nExample routing requests and expected parameter extraction:")
    
    examples = [
        {
            "request": "What's the weather in Seattle?",
            "expected_route": "weather",
            "expected_parameters": {
                "location": "Seattle",
                "timeframe": "current",
                "format": "brief"
            }
        },
        {
            "request": "How's the weather tomorrow in New York?",
            "expected_route": "weather", 
            "expected_parameters": {
                "location": "New York",
                "timeframe": "tomorrow",
                "format": "brief"
            }
        },
        {
            "request": "Give me a detailed weather forecast for 90210",
            "expected_route": "weather",
            "expected_parameters": {
                "location": "90210",
                "timeframe": "current",
                "format": "detailed"
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Request: \"{example['request']}\"")
        print(f"   Expected route: {example['expected_route']}")
        print(f"   Expected parameters:")
        for key, value in example['expected_parameters'].items():
            print(f"     - {key}: {value}")


def demonstrate_lifecycle_execution():
    """Demonstrate the lifecycle execution flow."""
    
    print("\n⚡ Hybrid Role Execution Flow")
    print("=" * 50)
    
    print("\nExecution phases for hybrid weather role:")
    print("1. 📥 Request received: \"What's the weather in Seattle?\"")
    print("2. 🎯 Enhanced routing extracts parameters:")
    print("   - location: Seattle")
    print("   - timeframe: current")
    print("   - format: brief")
    
    print("\n3. 🔄 Pre-processing phase:")
    print("   - fetch_weather_data(location='Seattle', timeframe='current')")
    print("     → Calls weather API")
    print("     → Returns: {weather_current: {...}, location_resolved: 'Seattle, WA'}")
    print("   - validate_location(location='Seattle')")
    print("     → Returns: {location_type: 'city_name', validation_status: 'valid'}")
    
    print("\n4. 🧠 LLM processing phase:")
    print("   - System prompt enhanced with pre-processed data")
    print("   - LLM interprets weather data instead of fetching it")
    print("   - Returns: \"It's currently 72°F and sunny in Seattle...\"")
    
    print("\n5. ✨ Post-processing phase:")
    print("   - format_for_tts(): \"72°F\" → \"72 degrees Fahrenheit\"")
    print("   - pii_scrubber(): Remove sensitive coordinates")
    print("   - audit_log(): Log interaction for compliance")
    
    print("\n6. 📤 Final result: \"It's currently 72 degrees Fahrenheit and sunny in Seattle...\"")


def demonstrate_performance_benefits():
    """Demonstrate performance benefits of hybrid roles."""
    
    print("\n📊 Performance Benefits")
    print("=" * 50)
    
    print("\nTraditional LLM Role Flow:")
    print("1. Route request → LLM call #1")
    print("2. Execute role → LLM call #2")
    print("3. LLM calls weather tool → External API call")
    print("4. LLM processes tool result → LLM call #3")
    print("Total: 3 LLM calls + 1 API call")
    
    print("\nHybrid Role Flow:")
    print("1. Enhanced routing + parameter extraction → LLM call #1")
    print("2. Pre-processing fetches data → External API call")
    print("3. LLM processes pre-fetched data → LLM call #2")
    print("4. Post-processing formats result → Local processing")
    print("Total: 2 LLM calls + 1 API call")
    
    print("\nBenefits:")
    print("✅ Fewer LLM calls (3 → 2)")
    print("✅ Faster execution (no tool call overhead)")
    print("✅ Better caching (structured parameters)")
    print("✅ Consistent formatting (post-processing)")
    print("✅ Enhanced security (PII scrubbing)")
    print("✅ Better monitoring (audit logging)")


def main():
    """Run the hybrid role demonstration."""
    try:
        demonstrate_hybrid_role_features()
        demonstrate_enhanced_routing()
        demonstrate_lifecycle_execution()
        demonstrate_performance_benefits()
        
        print("\n🎉 Hybrid Role Lifecycle Architecture Demo Complete!")
        print("\nNext steps:")
        print("- Try creating your own hybrid role using the migration guide")
        print("- Run the test suite: pytest tests/unit/test_hybrid_role_lifecycle.py")
        print("- Check the weather role example in roles/weather/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()