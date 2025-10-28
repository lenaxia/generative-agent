"""Role Registry Module

Manages dynamic loading and registration of roles from file-based definitions.
"""

import importlib.util
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml
from pydantic import ValidationError

from llm_provider.role_schema import validate_role_definition

logger = logging.getLogger(__name__)


@dataclass
class RoleDefinition:
    """Container for a complete role definition."""

    name: str
    config: dict[str, Any]
    custom_tools: list[Callable]
    shared_tools: dict[str, Callable]


class RoleRegistry:
    """Role registry for managing and accessing role definitions.

    Central registry for discovering, loading, and managing roles used in the system.
    """

    _global_registry = None

    def __init__(
        self,
        roles_directory: str = "roles",
        message_bus=None,
        intent_processor=None,
        workflow_engine=None,
    ):
        """Initialize the role registry.

        Args:
            roles_directory: Path to the roles directory
            message_bus: Optional MessageBus for dynamic event registration
            intent_processor: Optional IntentProcessor for intent handler registration
            workflow_engine: Optional WorkflowEngine for role execution context
        """
        self.roles_directory = Path(roles_directory)
        self.message_bus = message_bus
        self.intent_processor = intent_processor
        self._workflow_engine = workflow_engine

        # Enhanced role storage for hybrid architecture
        self.llm_roles: dict[str, RoleDefinition] = {}  # All roles are hybrid now
        self.role_types: dict[
            str, str
        ] = {}  # Role type mapping (kept for compatibility)

        # Hybrid role lifecycle support
        self.lifecycle_functions: dict[
            str, dict[str, Callable]
        ] = {}  # role_name -> {func_name: func}

        # Event handling support
        self._role_event_handlers: dict[
            str, dict[str, Callable]
        ] = {}  # role_name -> {event_type: handler}

        # Backward compatibility - keep existing interface
        self.roles: dict[
            str, RoleDefinition
        ] = self.llm_roles  # Alias for backward compatibility
        self.shared_tools: dict[str, Callable] = {}

        # Performance optimization: Track initialization state
        self._is_initialized = False
        self._fast_reply_roles_cache: Optional[list[RoleDefinition]] = None

        # Load shared tools first
        self._load_shared_tools()

        # Discover and load all roles
        self.refresh()

    @classmethod
    def get_global_registry(cls, roles_directory: str = "roles") -> "RoleRegistry":
        """Get the global role registry instance."""
        if cls._global_registry is None:
            cls._global_registry = cls(roles_directory)
        return cls._global_registry

    def set_intent_processor(self, intent_processor):
        """Set the intent processor for role intent handler registration.

        Args:
            intent_processor: IntentProcessor instance for registering role-specific intents
        """
        self.intent_processor = intent_processor
        logger.info("IntentProcessor set on RoleRegistry")

    def set_workflow_engine(self, workflow_engine):
        """Set WorkflowEngine reference for role execution context.

        Args:
            workflow_engine: WorkflowEngine instance for role execution context
        """
        self._workflow_engine = workflow_engine
        logger.info("WorkflowEngine set on RoleRegistry")

        # Re-register all single-file role intents
        self._register_all_single_file_role_intents()

    def _register_all_single_file_role_intents(self):
        """Re-register all single-file role intents with the IntentProcessor."""
        if not self.intent_processor:
            logger.warning("No IntentProcessor available for intent registration")
            return

        single_file_roles = self.get_single_file_roles()
        for role_name in single_file_roles:
            try:
                # Re-import and register intents for this role
                single_file_path = self.roles_directory / f"core_{role_name}.py"
                if single_file_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        f"roles.core_{role_name}", single_file_path
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        if hasattr(module, "register_role"):
                            registration = module.register_role()
                            if "intents" in registration:
                                self._register_single_file_role_intents(
                                    role_name, registration["intents"]
                                )

            except Exception as e:
                logger.error(f"Failed to re-register intents for {role_name}: {e}")

    def refresh(self):
        """Refresh role registry by discovering and loading all LLM roles from filesystem."""
        logger.debug("Refreshing role registry...")

        discovered_roles = self._discover_roles()
        logger.debug(
            f"Discovered {len(discovered_roles)} LLM roles: {discovered_roles}"
        )

        for role_name in discovered_roles:
            try:
                role_def = self._load_role(role_name)
                self.llm_roles[role_name] = role_def
                self.role_types[role_name] = "llm"
                logger.debug(
                    f"Loaded LLM role '{role_name}' with {len(role_def.custom_tools)} custom tools"
                )
            except Exception as e:
                logger.error(f"Failed to load role '{role_name}': {e}")

        # Mark as initialized and clear cache
        self._is_initialized = True
        self._fast_reply_roles_cache = None

        logger.debug(f"Role registry refreshed with {len(self.llm_roles)} hybrid roles")

    def initialize_once(self):
        """Initialize the role registry if not already initialized.

        This method provides performance optimization by avoiding repeated initialization.
        """
        if self._is_initialized:
            logger.debug("Role registry already initialized, skipping")
            return

        logger.info("Performing one-time role registry initialization...")
        self.refresh()
        logger.info("Role registry initialization completed")

    def _discover_roles(self) -> list[str]:
        """Discover all available role definitions (both multi-file and single-file)."""
        roles = []

        if not self.roles_directory.exists():
            logger.warning(f"Roles directory not found: {self.roles_directory}")
            return roles

        # Discover single-file roles FIRST (new LLM-safe pattern - prioritized)
        for role_file in self.roles_directory.glob("core_*.py"):
            role_name = role_file.stem.replace("core_", "")
            roles.append(role_name)
            logger.debug(f"Discovered single-file role: {role_name}")

        # Discover multi-file roles SECOND (legacy pattern - only if no single-file exists)
        for role_dir in self.roles_directory.iterdir():
            if role_dir.is_dir() and (role_dir / "definition.yaml").exists():
                # Skip shared_tools directory
                if role_dir.name != "shared_tools" and role_dir.name not in roles:
                    roles.append(role_dir.name)
                    logger.debug(f"Discovered multi-file role: {role_dir.name}")

        return roles

    def _load_role(self, role_name: str) -> RoleDefinition:
        """Enhanced role loading with support for both multi-file and single-file roles."""
        # Check for single-file role first (new LLM-safe pattern)
        single_file_path = self.roles_directory / f"core_{role_name}.py"

        if single_file_path.exists():
            return self._load_single_file_role(role_name, single_file_path)
        else:
            return self._load_multi_file_role(role_name)

    def _load_single_file_role(self, role_name: str, role_file: Path) -> RoleDefinition:
        """Load single-file role using register_role() function."""
        logger.debug(f"Loading single-file role: {role_name}")

        try:
            # Import the single-file role module
            spec = importlib.util.spec_from_file_location(
                f"roles.core_{role_name}", role_file
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {role_file}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get role registration
            if not hasattr(module, "register_role"):
                raise ValueError(
                    f"Single-file role {role_name} missing register_role() function"
                )

            registration = module.register_role()

            # Validate registration structure
            required_keys = ["config", "event_handlers", "tools", "intents"]
            for key in required_keys:
                if key not in registration:
                    raise ValueError(
                        f"Single-file role {role_name} registration missing '{key}' key"
                    )

            # Convert to RoleDefinition format
            role_tools_config = registration["config"].get(
                "tools", {"automatic": True, "shared": []}
            )
            config = {
                "role": registration["config"],
                "events": {
                    "subscribes": list(registration["event_handlers"].keys()),
                    "publishes": [],  # Single-file roles use intents instead of direct publishing
                },
                "tools": role_tools_config,  # Use tools config from role, not hardcoded
            }

            # Extract tools from registration
            custom_tools = registration.get("tools", [])

            # Register event handlers with MessageBus if available
            if self.message_bus:
                self._register_single_file_role_events(role_name, registration)

            # Register intent handlers if available
            if "intents" in registration:
                self._register_single_file_role_intents(
                    role_name, registration["intents"]
                )

            # Store pre-processors for Universal Agent access
            role_def = RoleDefinition(
                name=role_name,
                config=config,
                custom_tools=custom_tools,
                shared_tools=self.shared_tools,
            )

            # NEW: Store single-file role pre-processors
            if "pre_processors" in registration:
                role_def._pre_processors = registration["pre_processors"]
                logger.info(
                    f"Registered {len(registration['pre_processors'])} pre-processors for {role_name}"
                )

            # NEW: Load lifecycle functions from single-file role module
            lifecycle_functions = self._load_single_file_lifecycle_functions(
                role_name, module, registration["config"]
            )
            if lifecycle_functions:
                self.register_lifecycle_functions(role_name, lifecycle_functions)

            logger.debug(f"Successfully loaded single-file role: {role_name}")
            return role_def

        except Exception as e:
            logger.error(f"Failed to load single-file role {role_name}: {e}")
            raise

    def _load_multi_file_role(self, role_name: str) -> RoleDefinition:
        """Load multi-file role (legacy pattern)."""
        role_path = self.roles_directory / role_name
        definition_file = role_path / "definition.yaml"

        # Load role configuration
        with open(definition_file) as f:
            config = yaml.safe_load(f)

        # Validate role configuration against schema (validation only, don't transform)
        try:
            validate_role_definition(config)
            logger.debug(f"Role {role_name} passed schema validation")
        except ValidationError as e:
            logger.error(f"Role {role_name} failed schema validation: {e}")
            raise ValueError(
                f"Invalid role definition for '{role_name}': {e}\n"
                f"Please check {definition_file} for correct structure. "
                f"Common issues: 'tools' should be a dict like {{automatic: false, shared: []}}, not a list []"
            ) from e

        # Load custom tools if tools.py exists
        custom_tools = []
        tools_file = role_path / "tools.py"
        if tools_file.exists():
            custom_tools = self._load_custom_tools(tools_file)

        # Load lifecycle functions for all roles
        lifecycle_functions = self._load_lifecycle_functions(role_name)
        if lifecycle_functions:
            self.register_lifecycle_functions(role_name, lifecycle_functions)

        # Auto-register events if MessageBus is available
        if self.message_bus and "events" in config:
            self._register_role_events(role_name, config["events"])

        return RoleDefinition(
            name=role_name,
            config=config,
            custom_tools=custom_tools,
            shared_tools=self.shared_tools,
        )

    def _load_shared_tools(self):
        """Load shared tools from shared_tools directory."""
        shared_tools_dir = self.roles_directory / "shared_tools"

        if not shared_tools_dir.exists():
            logger.warning("Shared tools directory not found")
            return

        # Load tools from each shared tool file
        for tool_file in shared_tools_dir.glob("*.py"):
            if tool_file.name == "__init__.py":
                continue

            try:
                tools = self._load_tools_from_file(tool_file)
                for tool in tools:
                    if hasattr(tool, "_tool_name"):
                        self.shared_tools[tool._tool_name] = tool
                    else:
                        self.shared_tools[tool.__name__] = tool

                logger.info(f"Loaded {len(tools)} shared tools from {tool_file.name}")

            except Exception as e:
                logger.error(f"Failed to load shared tools from {tool_file}: {e}")

    def _load_custom_tools(self, tools_file: Path) -> list[Callable]:
        """Load all @tool functions from a role's tools.py file."""
        return self._load_tools_from_file(tools_file)

    def _load_tools_from_file(self, tools_file: Path) -> list[Callable]:
        """Load all @tool decorated functions from a Python file."""
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(
                f"tools_{tools_file.stem}", tools_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all @tool decorated functions
            tools = []
            for _name, obj in inspect.getmembers(module):
                # Check for StrandsAgent @tool decorator with multiple detection methods
                if callable(obj) and (
                    hasattr(obj, "_is_tool")  # Original check
                    or hasattr(obj, "__wrapped__")  # Decorator wrapper
                    or hasattr(obj, "_tool_name")  # StrandsAgent tool name
                    or getattr(obj, "__name__", "").startswith("tool_")  # Tool prefix
                    or hasattr(obj, "_strands_tool")  # StrandsAgent marker
                    or str(type(obj)).find("tool") != -1  # Tool in type name
                ):
                    tools.append(obj)

            return tools

        except Exception as e:
            logger.error(f"Failed to load tools from {tools_file}: {e}")
            return []

    def get_role(self, role_name: str) -> Optional[RoleDefinition]:
        """Get a role definition by name.

        Args:
            role_name: Name of the role

        Returns:
            RoleDefinition or None if not found
        """
        return self.roles.get(role_name)

    def get_role_summaries(self) -> list[dict[str, str]]:
        """Get summaries for all available roles.

        Returns:
            List of role summaries with name, description, and when_to_use
        """
        summaries = []
        for name, role_def in self.roles.items():
            role_config = role_def.config.get("role", {})
            summaries.append(
                {
                    "name": name,
                    "description": role_config.get("description", ""),
                    "when_to_use": role_config.get("when_to_use", ""),
                }
            )
        return summaries

    def list_roles(self) -> list[dict[str, Any]]:
        """List all available roles with detailed information.

        Returns:
            List of role information dicts
        """
        roles_info = []
        for name, role_def in self.roles.items():
            role_config = role_def.config.get("role", {})
            roles_info.append(
                {
                    "name": name,
                    "description": role_config.get("description", ""),
                    "version": role_config.get("version", "1.0.0"),
                    "author": role_config.get("author", ""),
                    "custom_tool_count": len(role_def.custom_tools),
                    "has_automatic_tools": role_def.config.get("tools", {}).get(
                        "automatic", False
                    ),
                    "shared_tools": role_def.config.get("tools", {}).get("shared", []),
                }
            )
        return roles_info

    def get_shared_tool(self, tool_name: str) -> Optional[Callable]:
        """Get a shared tool by name.

        Args:
            tool_name: Name of the shared tool

        Returns:
            Tool function or None if not found
        """
        return self.shared_tools.get(tool_name)

    def get_all_shared_tools(self) -> dict[str, Callable]:
        """Get all shared tools."""
        return self.shared_tools.copy()

    def get_role_llm_type(self, role_name: str) -> str:
        """Get the LLM type for a role from its definition.

        Args:
            role_name: Name of the role

        Returns:
            str: LLM type (WEAK, DEFAULT, STRONG) or DEFAULT if not specified
        """
        role_def = self.get_role(role_name)
        if not role_def:
            return "DEFAULT"

        # Get LLM type from role configuration
        role_config = role_def.config.get("role", {})
        llm_type = role_config.get("llm_type", "DEFAULT")

        # Validate LLM type
        valid_types = ["WEAK", "DEFAULT", "STRONG"]
        if llm_type not in valid_types:
            logger.warning(
                f"Invalid LLM type '{llm_type}' for role '{role_name}', using DEFAULT"
            )
            return "DEFAULT"

        return llm_type

    def get_all_role_llm_mappings(self) -> dict[str, str]:
        """Get LLM type mappings for all roles.

        Returns:
            dict: Mapping of role names to LLM types
        """
        mappings = {}
        for role_name in self.roles.keys():
            mappings[role_name] = self.get_role_llm_type(role_name)
        return mappings

    def validate_role(self, role_name: str) -> dict[str, Any]:
        """Validate a role definition.

        Args:
            role_name: Name of the role to validate

        Returns:
            Validation result with status and any errors
        """
        if role_name not in self.roles:
            return {
                "valid": False,
                "errors": [f"Role '{role_name}' not found"],
                "warnings": [],
            }

        role_def = self.roles[role_name]
        errors = []
        warnings = []

        # Check required fields
        role_config = role_def.config.get("role", {})
        required_fields = ["name", "description"]
        for field in required_fields:
            if not role_config.get(field):
                errors.append(f"Missing required field: role.{field}")

        # Check prompts
        prompts = role_def.config.get("prompts", {})
        if not prompts.get("system"):
            warnings.append("No system prompt defined")

        # Check tools configuration
        tools_config = role_def.config.get("tools", {})
        if (
            not tools_config.get("automatic")
            and not tools_config.get("shared")
            and not role_def.custom_tools
        ):
            warnings.append(
                "No tools configured (no automatic, shared, or custom tools)"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "role_name": role_name,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the role registry.

        Returns:
            Statistics about the role registry
        """
        total_custom_tools = sum(
            len(role_def.custom_tools) for role_def in self.roles.values()
        )
        roles_with_automatic = sum(
            1
            for role_def in self.roles.values()
            if role_def.config.get("tools", {}).get("automatic", False)
        )

        return {
            "total_roles": len(self.roles),
            "total_shared_tools": len(self.shared_tools),
            "total_custom_tools": total_custom_tools,
            "roles_with_automatic_tools": roles_with_automatic,
            "roles_with_custom_tools": sum(
                1 for role_def in self.roles.values() if role_def.custom_tools
            ),
            "average_tools_per_role": (
                total_custom_tools / len(self.roles) if self.roles else 0
            ),
        }

    # Enhanced methods for hybrid execution architecture
    # Programmatic role methods removed - everything is hybrid now

    def register_llm_role(self, name: str, definition: RoleDefinition):
        """Register an LLM-based role.

        Args:
            name: Role name
            definition: RoleDefinition instance
        """
        self.llm_roles[name] = definition
        self.role_types[name] = "llm"
        logger.info(f"Registered LLM role: {name}")

    def get_role_type(self, role_name: str) -> str:
        """Get the type of a role.

        Args:
            role_name: Name of the role

        Returns:
            "programmatic" or "llm" (defaults to "llm" for unknown roles)
        """
        return self.role_types.get(role_name, "llm")

    def get_all_roles(self) -> dict[str, str]:
        """Get all available roles and their types.

        Returns:
            Dict mapping role names to their execution types
        """
        return self.role_types.copy()

    # All programmatic role methods removed - everything is hybrid now

    # validate_programmatic_role method removed - everything is hybrid now

    def get_enhanced_statistics(self) -> dict[str, Any]:
        """Get enhanced statistics about the role registry.

        Returns:
            Enhanced statistics about the role registry
        """
        base_stats = self.get_statistics()

        # Add hybrid role statistics
        hybrid_stats = {
            "total_hybrid_roles": len(self.llm_roles),
            "role_type_distribution": {
                "hybrid": len(self.llm_roles),
            },
        }

        # Merge with base statistics
        enhanced_stats = {**base_stats, **hybrid_stats}
        enhanced_stats["total_all_roles"] = len(self.llm_roles)

        return enhanced_stats

    # Enhanced methods for hybrid execution architecture

    def get_role_parameters(self, role_name: str) -> dict[str, Any]:
        """Get parameter schema for a role for routing extraction."""
        role_def = self.get_role(role_name)
        if not role_def:
            return {}
        return role_def.config.get("parameters", {})

    def get_role_execution_type(self, role_name: str) -> str:
        """Get the execution type of a role.

        Args:
            role_name: Name of the role

        Returns:
            Always returns "hybrid" since all roles are hybrid now
        """
        # All roles are hybrid now - no need to check execution type
        return "hybrid"

    def register_lifecycle_functions(
        self, role_name: str, functions: dict[str, Callable]
    ):
        """Register lifecycle functions for a role."""
        self.lifecycle_functions[role_name] = functions
        logger.info(
            f"Registered {len(functions)} lifecycle functions for role: {role_name}"
        )

    def get_lifecycle_functions(self, role_name: str) -> dict[str, Callable]:
        """Get lifecycle functions for a role."""
        return self.lifecycle_functions.get(role_name, {})

    def _load_lifecycle_functions(self, role_name: str) -> dict[str, Callable]:
        """Load lifecycle functions from role's Python module."""
        # Load from roles/{role_name}/lifecycle.py if it exists
        lifecycle_file = self.roles_directory / role_name / "lifecycle.py"
        if lifecycle_file.exists():
            return self._load_functions_from_file(lifecycle_file)
        return {}

    def _load_single_file_lifecycle_functions(
        self, role_name: str, module, role_config: dict
    ) -> dict[str, Callable]:
        """Load lifecycle functions from single-file role module based on lifecycle configuration."""
        lifecycle_functions = {}

        # Get lifecycle configuration
        lifecycle_config = role_config.get("lifecycle", {})

        # Collect function names from pre-processing and post-processing configurations
        function_names = set()

        # Pre-processing functions
        pre_config = lifecycle_config.get("pre_processing", {})
        if pre_config.get("enabled", False):
            function_names.update(pre_config.get("functions", []))

        # Post-processing functions
        post_config = lifecycle_config.get("post_processing", {})
        if post_config.get("enabled", False):
            function_names.update(post_config.get("functions", []))

        # Load the specified functions from the module
        for func_name in function_names:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    lifecycle_functions[func_name] = func
                    logger.debug(
                        f"Loaded lifecycle function '{func_name}' for role '{role_name}'"
                    )
                else:
                    logger.warning(
                        f"Lifecycle function '{func_name}' is not callable in role '{role_name}'"
                    )
            else:
                logger.warning(
                    f"Lifecycle function '{func_name}' not found in role '{role_name}' module"
                )

        if lifecycle_functions:
            logger.info(
                f"Loaded {len(lifecycle_functions)} lifecycle functions for single-file role '{role_name}': {list(lifecycle_functions.keys())}"
            )

        return lifecycle_functions

    def _load_functions_from_file(self, lifecycle_file: Path) -> dict[str, Callable]:
        """Load all functions from a Python file."""
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(
                f"lifecycle_{lifecycle_file.parent.name}", lifecycle_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all callable functions (not classes or private methods)
            functions = {}
            for name, obj in inspect.getmembers(module):
                if (
                    callable(obj)
                    and not name.startswith("_")
                    and not inspect.isclass(obj)
                ):
                    functions[name] = obj

            logger.info(
                f"Loaded {len(functions)} lifecycle functions from {lifecycle_file}"
            )
            return functions

        except Exception as e:
            logger.error(
                f"Failed to load lifecycle functions from {lifecycle_file}: {e}"
            )
            return {}

    # Fast-reply role methods for fast-path routing
    def get_fast_reply_roles(self) -> list[RoleDefinition]:
        """Get roles marked for fast-reply execution.

        Uses caching for performance optimization.

        Returns:
            List[RoleDefinition]: Roles suitable for fast-reply execution
        """
        # Use cached result if available
        if self._fast_reply_roles_cache is not None:
            logger.debug("Returning cached fast-reply roles")
            return self._fast_reply_roles_cache

        # Compute and cache the result
        logger.debug("Computing fast-reply roles")
        fast_reply_roles = [
            role
            for role in self.llm_roles.values()
            if role.config.get("role", {}).get("fast_reply", False)
        ]

        self._fast_reply_roles_cache = fast_reply_roles
        logger.info(f"Cached {len(fast_reply_roles)} fast-reply roles")

        return fast_reply_roles

    def is_fast_reply_role(self, role_name: str) -> bool:
        """Check if a role supports fast replies.

        Args:
            role_name: Name of the role to check

        Returns:
            bool: True if role supports fast replies
        """
        role = self.get_role(role_name)
        if role is None:
            return False
        return role.config.get("role", {}).get("fast_reply", False)

    def get_fast_reply_role_summaries(self) -> list[dict[str, str]]:
        """Get summaries for roles that support fast replies.

        Returns:
            List of role summaries with name and description
        """
        fast_roles = self.get_fast_reply_roles()
        return [
            {
                "name": role.name,
                "description": role.config.get("role", {}).get(
                    "description", "No description available"
                ),
            }
            for role in fast_roles
        ]

    # Event registration and handling methods
    def _register_role_events(self, role_name: str, events_config: dict[str, Any]):
        """Register role's published and subscribed events.

        Args:
            role_name: Name of the role
            events_config: Events configuration from role definition
        """
        # Register published events
        if "publishes" in events_config:
            for event_config in events_config["publishes"]:
                self.message_bus.event_registry.register_event_type(
                    event_config["event_type"],
                    role_name,
                    event_config.get("data_schema", {}),
                    event_config.get("description", ""),
                )

        # Register subscriptions
        if "subscribes" in events_config:
            for subscription in events_config["subscribes"]:
                event_type = subscription["event_type"]
                handler_name = subscription["handler"]

                # Load handler function from role's lifecycle
                handler_func = self._load_role_handler(role_name, handler_name)

                if handler_func:
                    # Subscribe to MessageBus
                    self.message_bus.subscribe(role_name, event_type, handler_func)
                    self.message_bus.event_registry.register_subscription(
                        event_type, role_name
                    )

                    # Track handler for role
                    if role_name not in self._role_event_handlers:
                        self._role_event_handlers[role_name] = {}
                    self._role_event_handlers[role_name][event_type] = handler_func

    def _load_role_handler(
        self, role_name: str, handler_name: str
    ) -> Optional[Callable]:
        """Load event handler function from role's lifecycle module.

        Args:
            role_name: Name of the role
            handler_name: Name of the handler function

        Returns:
            Wrapped handler function or None if not found
        """
        try:
            # Import role's lifecycle module
            import importlib

            lifecycle_module = importlib.import_module(f"roles.{role_name}.lifecycle")

            # Get handler function
            if hasattr(lifecycle_module, handler_name):
                handler_func = getattr(lifecycle_module, handler_name)

                # Verify it's actually callable
                if not callable(handler_func):
                    logger.error(
                        f"Handler '{handler_name}' in {role_name}.lifecycle is not callable"
                    )
                    return None

                # Wrap handler to provide EventHandlerContext
                async def enhanced_handler(event_data):
                    # Create EventHandlerLLM utility for handlers
                    from common.event_handler_context import EventHandlerContext
                    from common.event_handler_llm import EventHandlerLLM

                    llm_utility = EventHandlerLLM(
                        llm_factory=self.message_bus.llm_factory,
                        event_context=(
                            event_data if isinstance(event_data, dict) else {}
                        ),
                    )

                    # Create context object with all dependencies
                    context = EventHandlerContext(
                        llm=llm_utility,
                        workflow_engine=self.message_bus.workflow_engine,
                        communication_manager=self.message_bus.communication_manager,
                        message_bus=self.message_bus,
                        execution_context=(
                            event_data.get("execution_context", {})
                            if isinstance(event_data, dict)
                            else {}
                        ),
                    )

                    # Call handler with enhanced signature (individual components as kwargs)
                    return await handler_func(
                        event_data,
                        llm=llm_utility,
                        workflow_engine=self.message_bus.workflow_engine,
                        communication_manager=self.message_bus.communication_manager,
                        context=context,
                    )

                return enhanced_handler
            else:
                logger.error(
                    f"Handler '{handler_name}' not found in {role_name}.lifecycle"
                )
                return None

        except ImportError as e:
            logger.error(
                f"Could not import lifecycle module for role '{role_name}': {e}"
            )
            return None

    def get_role_events_info(self, role_name: str) -> dict[str, Any]:
        """Get event information for a specific role.

        Args:
            role_name: Name of the role

        Returns:
            Dictionary with role's event information
        """
        role_def = self.get_role(role_name)
        if not role_def or "events" not in role_def:
            return {"publishes": [], "subscribes": [], "handlers": []}

        return {
            "publishes": role_def["events"].get("publishes", []),
            "subscribes": role_def["events"].get("subscribes", []),
            "handlers": list(self._role_event_handlers.get(role_name, {}).keys()),
        }

    def _register_single_file_role_events(
        self, role_name: str, registration: dict[str, Any]
    ):
        """Register event handlers for single-file roles."""
        try:
            event_handlers = registration.get("event_handlers", {})

            for event_type, handler_func in event_handlers.items():
                # Register with MessageBus
                if self.message_bus:
                    self.message_bus.subscribe(role_name, event_type, handler_func)
                    logger.debug(
                        f"Registered event handler {event_type} for single-file role {role_name}"
                    )

                # Store in local registry
                if role_name not in self._role_event_handlers:
                    self._role_event_handlers[role_name] = {}
                self._role_event_handlers[role_name][event_type] = handler_func

        except Exception as e:
            logger.error(
                f"Failed to register events for single-file role {role_name}: {e}"
            )

    def _register_single_file_role_intents(
        self, role_name: str, intents: dict[type, Callable]
    ):
        """Register intent handlers for single-file roles."""
        try:
            for intent_type, handler_func in intents.items():
                logger.debug(
                    f"Registered intent handler {intent_type.__name__} for single-file role {role_name}"
                )

                # Register with IntentProcessor if available
                if self.intent_processor:
                    self.intent_processor.register_role_intent_handler(
                        intent_type, handler_func, role_name
                    )
                    logger.info(
                        f"Intent handler {intent_type.__name__} registered with IntentProcessor"
                    )
                else:
                    logger.debug(
                        f"IntentProcessor not available, intent handler {intent_type.__name__} stored for later registration"
                    )

        except Exception as e:
            logger.error(
                f"Failed to register intents for single-file role {role_name}: {e}"
            )

    def get_single_file_roles(self) -> list[str]:
        """Get list of single-file roles."""
        single_file_roles = []

        if not self.roles_directory.exists():
            return single_file_roles

        for role_file in self.roles_directory.glob("core_*.py"):
            role_name = role_file.stem.replace("core_", "")
            single_file_roles.append(role_name)

        return single_file_roles

    def get_multi_file_roles(self) -> list[str]:
        """Get list of multi-file roles (legacy pattern)."""
        multi_file_roles = []

        if not self.roles_directory.exists():
            return multi_file_roles

        for role_dir in self.roles_directory.iterdir():
            if role_dir.is_dir() and (role_dir / "definition.yaml").exists():
                if role_dir.name != "shared_tools":
                    multi_file_roles.append(role_dir.name)

        return multi_file_roles

    def get_role_migration_status(self) -> dict[str, Any]:
        """Get status of role migration to single-file architecture."""
        single_file_roles = self.get_single_file_roles()
        multi_file_roles = self.get_multi_file_roles()

        return {
            "single_file_roles": single_file_roles,
            "multi_file_roles": multi_file_roles,
            "total_roles": len(single_file_roles) + len(multi_file_roles),
            "migration_progress": (
                len(single_file_roles)
                / (len(single_file_roles) + len(multi_file_roles))
                if (len(single_file_roles) + len(multi_file_roles)) > 0
                else 0
            ),
            "migrated_count": len(single_file_roles),
            "remaining_count": len(multi_file_roles),
        }
