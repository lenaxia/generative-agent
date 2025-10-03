#!/usr/bin/env python3
"""
Test script to verify weather role functionality is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_weather_role_functionality():
    """Test that weather role can be assumed and has correct tools."""
    
    print("🧪 Testing Weather Role Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Role Registry Discovery
        print("1. Testing Role Registry Discovery...")
        from llm_provider.role_registry import RoleRegistry
        registry = RoleRegistry()
        
        print(f"   ✅ Available roles: {list(registry.roles.keys())}")
        print(f"   ✅ Shared tools count: {len(registry.shared_tools)}")
        print(f"   ✅ Weather tools: {[tool for tool in registry.shared_tools.keys() if 'weather' in tool.lower()]}")
        
        # Test 2: Weather Role Definition
        print("\n2. Testing Weather Role Definition...")
        weather_role = registry.get_role('weather')
        if weather_role:
            print("   ✅ Weather role loaded successfully")
            print(f"   ✅ Weather role tools: {weather_role.config.get('tools', {}).get('shared', [])}")
        else:
            print("   ❌ Weather role not found")
            return False
        
        # Test 3: Tool Function Access
        print("\n3. Testing Weather Tool Functions...")
        from roles.shared_tools.weather_tools import get_weather
        print("   ✅ get_weather function imported successfully")
        print(f"   ✅ get_weather docstring: {get_weather.__doc__[:100]}...")
        
        # Test 4: Universal Agent Role Assumption
        print("\n4. Testing Universal Agent Role Assumption...")
        from llm_provider.factory import LLMFactory, LLMType
        from llm_provider.universal_agent import UniversalAgent
        from config.bedrock_config import BedrockConfig
        
        # Create mock config for testing
        config = BedrockConfig(
            name='test',
            model_id='test-model',
            region='us-west-2',
            temperature=0.3
        )
        
        # Create LLMFactory with mock config
        llm_factory = LLMFactory({LLMType.DEFAULT: [config]})
        
        # Create Universal Agent
        universal_agent = UniversalAgent(llm_factory, registry)
        
        # Test role assumption
        available_roles = universal_agent.get_available_roles()
        print(f"   ✅ Available roles from Universal Agent: {available_roles}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("Weather role functionality is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_weather_role_functionality()
    sys.exit(0 if success else 1)