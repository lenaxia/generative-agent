"""Tool Registry Module

Manages dynamic loading and registration of tools from domain-based definitions.
Central registry for ALL system capabilities (tools).
"""

import importlib.util
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Tool registry for managing and accessing all system tools.

    Central registry for discovering, loading, and managing tools from domain modules.
    Tools are organized by domain (category) and registered with fully qualified names.
    """

    def __init__(self):
        """Initialize the tool registry."""
        # Tool storage: {fully_qualified_name: Tool}
        # e.g., "weather.get_forecast": Tool object
        self._tools: dict[str, Any] = {}

        # Category storage: {category: [tool_names]}
        # e.g., "weather": ["weather.get_forecast", "weather.get_current"]
        self._categories: dict[str, list[str]] = {}

        # Initialization state
        self._loaded = False

        logger.debug("ToolRegistry initialized (empty)")

    async def initialize(self, config: dict[str, Any], providers: Any):
        """Load all tools from domain modules.

        Args:
            config: System configuration
            providers: Provider instances for tool dependencies

        This method discovers and loads tools from:
        - Domain modules with roles (e.g., roles/weather/tools.py)
        - Utility domain modules (e.g., roles/memory/tools.py)
        """
        if self._loaded:
            logger.warning("ToolRegistry already loaded, skipping initialization")
            return

        logger.info("Initializing ToolRegistry...")

        # Domain modules with dedicated roles
        domain_modules = [
            ("weather", "roles.weather.tools", getattr(providers, "weather", None)),
            ("calendar", "roles.calendar.tools", getattr(providers, "calendar", None)),
            ("timer", "roles.timer.tools", getattr(providers, "redis", None)),
            (
                "smart_home",
                "roles.smart_home.tools",
                getattr(providers, "home_assistant", None),
            ),
        ]

        # Core infrastructure tools (moved to tools/core/)
        core_tool_modules = [
            ("memory", "tools.core.memory", getattr(providers, "memory", None)),
            (
                "notification",
                "tools.core.notification",
                getattr(providers, "communication", None),
            ),
            (
                "summarization",
                "tools.core.summarization",
                getattr(providers, "llm_factory", None),
            ),
        ]

        # Domain tool modules (tools associated with domain roles)
        domain_tool_modules = [
            ("search", "roles.search.tools", getattr(providers, "search", None)),
            ("planning", "roles.planning.tools", getattr(providers, "planning", None)),
        ]

        # Load all modules (domain roles, core tools, and domain tool modules)
        all_tool_modules = domain_modules + core_tool_modules + domain_tool_modules
        for category, module_path, provider in all_tool_modules:
            try:
                if provider is None:
                    logger.warning(
                        f"Provider for {category} is None, skipping tool loading"
                    )
                    continue

                tools = await self._load_tools_from_module(module_path, provider)
                self._register_tools(category, tools)

                logger.info(
                    f"Loaded {len(tools)} tools from {category} domain: "
                    f"{[f'{category}.{t.name}' if hasattr(t, 'name') else f'{category}.{t.__name__}' for t in tools]}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to load tools from {category}: {e}", exc_info=True
                )

        self._loaded = True
        logger.info(
            f"ToolRegistry initialized with {len(self._tools)} tools "
            f"across {len(self._categories)} categories"
        )

    async def _load_tools_from_module(
        self, module_path: str, provider: Any
    ) -> list[Any]:
        """Load tools from a module by calling create_*_tools() function.

        Args:
            module_path: Python module path (e.g., 'roles.weather.tools')
            provider: Provider instance to pass to tool creation function

        Returns:
            List of tool objects (Strands Tools)
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Find the tool creation function (e.g., create_weather_tools)
            # Pattern: create_<category>_tools
            tool_creation_func = None
            for name, obj in inspect.getmembers(module):
                if (
                    name.startswith("create_")
                    and name.endswith("_tools")
                    and callable(obj)
                ):
                    tool_creation_func = obj
                    break

            if tool_creation_func is None:
                logger.warning(
                    f"No create_*_tools() function found in {module_path}, "
                    f"trying to find any callable functions"
                )
                # Fallback: look for any tool functions directly
                return self._load_tools_from_module_fallback(module)

            # Call the creation function with provider
            if inspect.iscoroutinefunction(tool_creation_func):
                tools = await tool_creation_func(provider)
            else:
                tools = tool_creation_func(provider)

            if not isinstance(tools, list):
                logger.warning(
                    f"Tool creation function in {module_path} did not return a list, got {type(tools)}"
                )
                return []

            return tools

        except ImportError as e:
            logger.error(f"Could not import {module_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading tools from {module_path}: {e}", exc_info=True)
            return []

    def _load_tools_from_module_fallback(self, module: Any) -> list[Any]:
        """Fallback: Load tool functions directly from module.

        This is used when no create_*_tools() function is found.
        """
        tools = []
        for name, obj in inspect.getmembers(module):
            # Check if it's a tool (various detection methods)
            if callable(obj) and (
                hasattr(obj, "_is_tool")
                or hasattr(obj, "_tool_name")
                or hasattr(obj, "_strands_tool")
            ):
                tools.append(obj)

        logger.info(f"Loaded {len(tools)} tools via fallback method")
        return tools

    def _register_tools(self, category: str, tools: list[Any]):
        """Register tools under a category.

        Args:
            category: Tool category (e.g., 'weather', 'calendar')
            tools: List of tool objects
        """
        if category not in self._categories:
            self._categories[category] = []

        for tool in tools:
            # Get tool name (handle different tool object structures)
            tool_name = self._extract_tool_name(tool)

            # Create fully qualified name
            full_name = f"{category}.{tool_name}"

            # Check for conflicts
            if full_name in self._tools:
                logger.warning(
                    f"Tool naming conflict: {full_name} already registered, overwriting"
                )

            # Register tool
            self._tools[full_name] = tool
            self._categories[category].append(full_name)

            logger.debug(f"Registered tool: {full_name}")

    def _extract_tool_name(self, tool: Any) -> str:
        """Extract tool name from tool object.

        Handles different tool object types (Strands Tool, functions, etc.)
        """
        # Try various attributes where tool name might be stored
        if hasattr(tool, "name"):
            return tool.name
        if hasattr(tool, "_tool_name"):
            return tool._tool_name
        if hasattr(tool, "__name__"):
            return tool.__name__

        # Fallback to string representation
        return str(tool)

    def get_tool(self, tool_name: str) -> Any | None:
        """Get a specific tool by fully qualified name.

        Args:
            tool_name: Fully qualified tool name (e.g., 'weather.get_forecast')

        Returns:
            Tool object or None if not found
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            logger.warning(f"Tool not found: {tool_name}")
        return tool

    def get_tools(self, tool_names: list[str]) -> list[Any]:
        """Get multiple tools by name.

        Args:
            tool_names: List of fully qualified tool names

        Returns:
            List of tool objects (skips tools not found)
        """
        tools = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool is not None:
                tools.append(tool)
            else:
                logger.warning(f"Requested tool not found: {name}")

        return tools

    def get_all_tools(self) -> list[Any]:
        """Get all registered tools.

        Returns:
            List of all tool objects
        """
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> list[Any]:
        """Get all tools in a specific category.

        Args:
            category: Category name (e.g., 'weather')

        Returns:
            List of tool objects in that category
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names]

    def get_category_names(self) -> list[str]:
        """Get list of all tool categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def format_for_llm(self) -> str:
        """Format all tools for LLM consumption (meta-planning).

        Returns a formatted string listing all tools with descriptions,
        organized by category. This is used by the meta-planning agent
        to select tools for dynamic agent creation.

        Returns:
            Formatted string of all tools
        """
        output = [f"AVAILABLE TOOLS ({len(self._tools)} total)\n"]

        for category in sorted(self._categories.keys()):
            tools = self.get_tools_by_category(category)
            output.append(f"\n{category.upper()} ({len(tools)} tools):")

            for tool in tools:
                tool_name = self._extract_tool_name(tool)
                full_name = f"{category}.{tool_name}"

                # Get tool description
                description = self._extract_tool_description(tool)

                output.append(f"  • {full_name}")
                if description:
                    output.append(f"    {description}")

                # Get parameters if available
                params = self._extract_parameters(tool)
                if params:
                    output.append(f"    Parameters: {', '.join(params)}")

        return "\n".join(output)

    def _extract_tool_description(self, tool: Any) -> str:
        """Extract description from tool object."""
        # Try various attributes
        if hasattr(tool, "description"):
            return tool.description
        if hasattr(tool, "_description"):
            return tool._description
        if hasattr(tool, "__doc__") and tool.__doc__:
            # Use first line of docstring
            return tool.__doc__.strip().split("\n")[0]

        return ""

    def _extract_parameters(self, tool: Any) -> list[str]:
        """Extract parameter names from tool."""
        # Try Strands tool schema
        if hasattr(tool, "parameters"):
            if isinstance(tool.parameters, dict):
                return list(tool.parameters.keys())

        # Try function signature
        if callable(tool):
            try:
                sig = inspect.signature(tool)
                params = [
                    name
                    for name, param in sig.parameters.items()
                    if name not in ["self", "cls"]
                ]
                return params
            except (ValueError, TypeError):
                pass

        return []

    def get_tool_summary(self) -> dict[str, Any]:
        """Get summary of tool registry for inspection/debugging.

        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_tools": len(self._tools),
            "total_categories": len(self._categories),
            "categories": {
                category: len(tool_names)
                for category, tool_names in self._categories.items()
            },
            "loaded": self._loaded,
        }

    def is_loaded(self) -> bool:
        """Check if registry has been loaded."""
        return self._loaded
