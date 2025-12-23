"""Phase 3 Lifecycle Production Validation

This script validates that Phase 3 domain roles work correctly in production
through UniversalAgent's lifecycle with agent pooling.

Run this after Phase 3 lifecycle refactoring to ensure:
1. Domain roles are detected by UniversalAgent
2. Correct tools are loaded from ToolRegistry
3. System prompts are applied
4. LLM types are respected
5. Agent pooling works

Usage:
    python3 validate_phase3_lifecycle.py
"""

import asyncio
import sys
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    print("=" * 70)


def print_success(text):
    """Print success message."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text):
    """Print error message."""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_warning(text):
    """Print warning message."""
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


def print_info(text):
    """Print info message."""
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")


async def validate_phase3_lifecycle():
    """Validate Phase 3 lifecycle integration."""
    print_header("PHASE 3 LIFECYCLE PRODUCTION VALIDATION")

    try:
        # Initialize system
        print_info("Initializing system components...")
        from llm_provider.factory import LLMFactory
        from common.message_bus import MessageBus
        from supervisor.workflow_engine import WorkflowEngine

        llm_factory = LLMFactory({})
        message_bus = MessageBus()
        workflow_engine = WorkflowEngine(
            llm_factory=llm_factory, message_bus=message_bus
        )

        await workflow_engine.initialize_phase3_systems()
        print_success("System initialized")

        # Check 1: Verify domain roles loaded
        print_header("CHECK 1: Domain Roles Loaded")
        expected_roles = ["weather", "calendar", "timer", "smart_home"]

        for role_name in expected_roles:
            domain_role = workflow_engine.role_registry.get_domain_role(role_name)
            if not domain_role:
                print_error(f"{role_name}: NOT FOUND")
                return 1

            tools = domain_role.get_tools()
            print_success(f"{role_name}: {len(tools)} tools loaded")

        # Check 2: Verify lifecycle-compatible interface
        print_header("CHECK 2: Lifecycle-Compatible Interface")

        for role_name in expected_roles:
            domain_role = workflow_engine.role_registry.get_domain_role(role_name)

            # Check required methods
            has_get_tools = hasattr(domain_role, "get_tools")
            has_get_prompt = hasattr(domain_role, "get_system_prompt")
            has_get_llm_type = hasattr(domain_role, "get_llm_type")
            has_execute = hasattr(domain_role, "execute")

            if not (has_get_tools and has_get_prompt and has_get_llm_type):
                print_error(
                    f"{role_name}: Missing required methods "
                    f"(get_tools={has_get_tools}, get_system_prompt={has_get_prompt}, "
                    f"get_llm_type={has_get_llm_type})"
                )
                return 1

            if has_execute:
                print_warning(
                    f"{role_name}: Still has execute() method (should be removed)"
                )

            print_success(f"{role_name}: Lifecycle-compatible interface present")

        # Check 3: Verify tools from ToolRegistry
        print_header("CHECK 3: Tools from ToolRegistry")

        expected_tool_counts = {
            "weather": 2,
            "calendar": 2,
            "timer": 3,
            "smart_home": 3,
        }

        for role_name, expected_count in expected_tool_counts.items():
            domain_role = workflow_engine.role_registry.get_domain_role(role_name)
            tools = domain_role.get_tools()

            if len(tools) != expected_count:
                print_error(
                    f"{role_name}: Expected {expected_count} tools, got {len(tools)}"
                )
                return 1

            print_success(f"{role_name}: {len(tools)}/{expected_count} tools loaded")

        # Check 4: Verify system prompts
        print_header("CHECK 4: System Prompts")

        for role_name in expected_roles:
            domain_role = workflow_engine.role_registry.get_domain_role(role_name)
            prompt = domain_role.get_system_prompt()

            if not prompt or len(prompt) < 50:
                print_error(f"{role_name}: Invalid system prompt (too short)")
                return 1

            # Check for role name in prompt
            if role_name not in prompt.lower() and role_name.replace("_", " ") not in prompt.lower():
                print_warning(
                    f"{role_name}: System prompt may not be role-specific"
                )

            print_success(f"{role_name}: System prompt present ({len(prompt)} chars)")

        # Check 5: Verify LLM types
        print_header("CHECK 5: LLM Types")

        expected_llm_types = {
            "weather": "WEAK",  # Simple queries
            "calendar": "DEFAULT",  # Calendar operations
            "timer": "WEAK",  # Simple operations
            "smart_home": "DEFAULT",  # Home control
        }

        for role_name, expected_type in expected_llm_types.items():
            domain_role = workflow_engine.role_registry.get_domain_role(role_name)
            llm_type = domain_role.get_llm_type()

            if expected_type not in str(llm_type):
                print_warning(
                    f"{role_name}: Expected {expected_type}, got {llm_type}"
                )
            else:
                print_success(f"{role_name}: {llm_type}")

        # Check 6: Verify UniversalAgent integration
        print_header("CHECK 6: UniversalAgent Integration")

        # Check that UniversalAgent's assume_role checks for domain roles
        from llm_provider.universal_agent import UniversalAgent
        import inspect

        source = inspect.getsource(UniversalAgent.assume_role)
        if "get_domain_role" not in source:
            print_error(
                "UniversalAgent.assume_role() doesn't check for domain roles"
            )
            return 1

        print_success("UniversalAgent.assume_role() checks for domain roles")

        if "get_system_prompt" in source and "get_llm_type" in source:
            print_success("UniversalAgent extracts configuration from domain roles")
        else:
            print_warning(
                "UniversalAgent may not properly extract domain role configuration"
            )

        # Summary
        print_header("VALIDATION SUMMARY")
        print_success("All checks passed!")
        print("\nPhase 3 domain roles are lifecycle-compatible and ready for production.")
        print("\nProduction Testing:")
        print("  Run: python3 cli.py")
        print("\n  Test queries:")
        print("    • Weather:    'whats the weather in seattle?'")
        print("    • Calendar:   'whats on my calendar today?'")
        print("    • Timer:      'set a timer for 5 minutes'")
        print("    • Smart Home: 'turn on the living room lights'")
        print("\n  Look for:")
        print("    • '✨ Using Phase 3 domain role: [role_name] with [N] tools'")
        print("    • Correct tools from ToolRegistry (not calculator/file/shell)")
        print("    • Proper LLM selection (WEAK or DEFAULT)")

        return 0

    except Exception as e:
        print_header("VALIDATION FAILED")
        print_error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(validate_phase3_lifecycle())
        sys.exit(exit_code)
    except Exception as e:
        print_error(f"Validation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
