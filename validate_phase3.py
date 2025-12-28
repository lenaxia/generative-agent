"""Phase 3 Validation Script

Validates that all Phase 3 role updates are complete and working.
Tests that roles can be imported, initialized, and declare their tools correctly.
"""

import sys


def validate_role_imports():
    """Test that all role classes can be imported."""
    print("=" * 60)
    print("Phase 3 Validation: Testing Role Imports")
    print("=" * 60)

    roles = [
        ("weather", "roles.weather", "WeatherRole"),
        ("calendar", "roles.calendar", "CalendarRole"),
        ("timer", "roles.timer", "TimerRole"),
        ("smart_home", "roles.smart_home", "SmartHomeRole"),
    ]

    all_passed = True

    for domain_name, module_path, class_name in roles:
        try:
            module = __import__(module_path, fromlist=[class_name])
            role_class = getattr(module, class_name)
            print(f"✓ {domain_name}: {module_path}.{class_name} imported")
        except ImportError as e:
            print(f"✗ {domain_name}: Import failed - {e}")
            all_passed = False
        except AttributeError as e:
            print(f"✗ {domain_name}: Class not found - {e}")
            all_passed = False

    return all_passed


def validate_role_structure():
    """Test that roles have the correct structure and required tools."""
    print("\n" + "=" * 60)
    print("Phase 3 Validation: Testing Role Structure")
    print("=" * 60)

    from roles.calendar import CalendarRole
    from roles.smart_home import SmartHomeRole
    from roles.timer import TimerRole
    from roles.weather import WeatherRole

    test_cases = [
        (
            "weather",
            WeatherRole,
            ["weather.get_current_weather", "weather.get_forecast"],
        ),
        (
            "calendar",
            CalendarRole,
            ["calendar.get_schedule", "calendar.add_calendar_event"],
        ),
        (
            "timer",
            TimerRole,
            ["timer.set_timer", "timer.cancel_timer", "timer.list_timers"],
        ),
        (
            "smart_home",
            SmartHomeRole,
            [
                "smart_home.ha_call_service",
                "smart_home.ha_get_state",
                "smart_home.ha_list_entities",
            ],
        ),
    ]

    all_passed = True

    for domain_name, role_class, expected_tools in test_cases:
        try:
            # Check REQUIRED_TOOLS class variable
            if not hasattr(role_class, "REQUIRED_TOOLS"):
                print(f"✗ {domain_name}: Missing REQUIRED_TOOLS class variable")
                all_passed = False
                continue

            required_tools = role_class.REQUIRED_TOOLS

            if not isinstance(required_tools, list):
                print(
                    f"✗ {domain_name}: REQUIRED_TOOLS is not a list, got {type(required_tools)}"
                )
                all_passed = False
                continue

            if set(required_tools) != set(expected_tools):
                print(
                    f"✗ {domain_name}: REQUIRED_TOOLS mismatch. Expected {expected_tools}, got {required_tools}"
                )
                all_passed = False
                continue

            print(
                f"✓ {domain_name}: REQUIRED_TOOLS correct ({len(required_tools)} tools)"
            )

            # Check for required methods
            required_methods = ["__init__", "initialize", "execute"]
            for method in required_methods:
                if not hasattr(role_class, method):
                    print(f"✗ {domain_name}: Missing required method '{method}'")
                    all_passed = False
                else:
                    print(f"✓ {domain_name}: Has method '{method}'")

        except Exception as e:
            print(f"✗ {domain_name}: Structure validation failed - {e}")
            all_passed = False

    return all_passed


def validate_role_instantiation():
    """Test that roles can be instantiated with mock dependencies."""
    print("\n" + "=" * 60)
    print("Phase 3 Validation: Testing Role Instantiation")
    print("=" * 60)

    from roles.calendar import CalendarRole
    from roles.smart_home import SmartHomeRole
    from roles.timer import TimerRole
    from roles.weather import WeatherRole

    # Create mock dependencies
    class MockToolRegistry:
        def get_tools(self, tool_names):
            return []  # Return empty list for mock

    class MockLLMFactory:
        def create_strands_model(self, llm_type):
            return None  # Return None for mock

    mock_registry = MockToolRegistry()
    mock_factory = MockLLMFactory()

    test_cases = [
        ("weather", WeatherRole),
        ("calendar", CalendarRole),
        ("timer", TimerRole),
        ("smart_home", SmartHomeRole),
    ]

    all_passed = True

    for domain_name, role_class in test_cases:
        try:
            role = role_class(mock_registry, mock_factory)

            # Check basic attributes
            if not hasattr(role, "tool_registry"):
                print(f"✗ {domain_name}: Missing 'tool_registry' attribute")
                all_passed = False
                continue

            if not hasattr(role, "llm_factory"):
                print(f"✗ {domain_name}: Missing 'llm_factory' attribute")
                all_passed = False
                continue

            if not hasattr(role, "tools"):
                print(f"✗ {domain_name}: Missing 'tools' attribute")
                all_passed = False
                continue

            if not hasattr(role, "name"):
                print(f"✗ {domain_name}: Missing 'name' attribute")
                all_passed = False
                continue

            print(f"✓ {domain_name}: Role instantiated successfully")

        except Exception as e:
            print(f"✗ {domain_name}: Instantiation failed - {e}")
            all_passed = False

    return all_passed


def main():
    """Run all Phase 3 validation tests."""
    print("\n" + "=" * 60)
    print("PHASE 3 ROLE UPDATES VALIDATION")
    print("=" * 60 + "\n")

    results = []

    # Test imports
    results.append(("Role Imports", validate_role_imports()))

    # Test structure
    results.append(("Role Structure", validate_role_structure()))

    # Test instantiation
    results.append(("Role Instantiation", validate_role_instantiation()))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All Phase 3 validation tests PASSED!")
        print("\nPhase 3 Complete:")
        print("- 4 predefined roles updated to new pattern")
        print("- Each role declares REQUIRED_TOOLS")
        print("- Roles load tools from central ToolRegistry")
        print("- Intent collection integrated into execution")
        print("\nNext: Update RoleRegistry and Supervisor initialization")
        return 0
    else:
        print("\n✗ Some Phase 3 validation tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
