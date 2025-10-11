"""
EventHandlerLLM utility for simplified LLM access in event handlers.

This module provides a clean interface for event handlers to make direct LLM calls
for parsing requests, extracting parameters, or generating workflow instructions.
It's intentionally minimal - just LLM calls with context, no agents, no tools, no roles.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EventHandlerLLM:
    """Simple LLM utility for event handlers - direct LLM calls only.

    This class provides a clean interface for event handlers to make direct LLM calls
    for parsing requests, extracting parameters, or generating workflow instructions.
    It's intentionally minimal - just LLM calls with context, no agents, no tools, no roles.

    The handler is responsible for using the LLM output to create workflows or take actions.
    """

    def __init__(self, llm_factory, event_context: dict[str, Any]):
        """Initialize LLM utility.

        Args:
            llm_factory: LLMFactory instance for creating LLM models
            event_context: Full event data including execution_context
        """
        self.llm_factory = llm_factory
        self.event_context = event_context
        self._execution_context = event_context.get("execution_context", {})

    async def invoke(
        self, prompt: str, model_type: str = "WEAK", context: dict[str, Any] = None
    ) -> str:
        """Direct LLM invocation - no roles, no tools, just LLM.

        Args:
            prompt: The prompt to send to the LLM
            model_type: LLM model strength ("WEAK", "DEFAULT", "STRONG")
            context: Additional context (merged with event context)

        Returns:
            Raw LLM response string

        Example:
            response = await llm.invoke("Parse this timer action: 'turn on lights'")
            complex_parse = await llm.invoke("Complex analysis...", model_type="STRONG")
        """
        # Merge event context with provided context for the prompt
        merged_context = {**self._execution_context}
        if context:
            merged_context.update(context)

        # Add context to prompt if provided
        if merged_context:
            context_str = f"\nContext: {json.dumps(merged_context, indent=2)}\n"
            prompt = f"{prompt}{context_str}"

        # Create LLM model directly (no agent wrapper)
        from llm_provider.factory import LLMType

        llm_model = self.llm_factory.create_strands_model(LLMType[model_type])

        # Make direct LLM call
        return await llm_model.invoke(prompt)

    async def parse_json(
        self, prompt: str, model_type: str = "WEAK", context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Parse JSON response from LLM with error handling.

        Args:
            prompt: Prompt that should return JSON
            model_type: LLM model strength to use
            context: Additional context

        Returns:
            Parsed JSON dict, or empty dict if parsing fails

        Example:
            result = await llm.parse_json('''
                Parse this timer action: "turn on the lights"
                Return: {"action": "smart_home", "device": "lights", "room": "bedroom"}
            ''')
        """
        try:
            response = await self.invoke(prompt, model_type, context)
            # Try to extract JSON from response if it's wrapped in text
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                response = response[json_start:json_end]

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response was: {response}")
            return {}
        except Exception as e:
            logger.error(f"Error in parse_json: {e}")
            return {}

    async def quick_decision(self, question: str, options: list[str] = None) -> str:
        """Make a quick decision using LLM.

        Args:
            question: Question to ask the LLM
            options: Optional list of valid options

        Returns:
            LLM decision as string

        Example:
            decision = await llm.quick_decision(
                "Should this timer create a workflow or just send a notification?",
                ["workflow", "notification"]
            )
        """
        options_text = f" Choose from: {', '.join(options)}" if options else ""
        prompt = f"{question}{options_text}"

        response = await self.invoke(prompt, model_type="WEAK")  # Use fast WEAK model

        # If options provided, try to match response to valid option
        if options:
            response_lower = response.lower()
            for option in options:
                if option.lower() in response_lower:
                    return option

        return response.strip()

    def get_context(self, key: str = None) -> Any:
        """Get execution context data.

        Args:
            key: Specific context key to retrieve, or None for full context

        Returns:
            Context value or full context dict

        Example:
            user_id = llm.get_context("user_id")
            room = llm.get_context("device_context.room")
            full_context = llm.get_context()
        """
        if key is None:
            return self._execution_context

        # Support dot notation for nested keys
        if "." in key:
            keys = key.split(".")
            value = self._execution_context
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            return value

        return self._execution_context.get(key)

    def get_original_request(self) -> str:
        """Get the original user request that triggered this event."""
        return self.event_context.get("original_request", "")

    def get_timer_id(self) -> str:
        """Get timer ID if this is a timer event."""
        return self.event_context.get("timer_id", "")

    def get_user_id(self) -> str:
        """Get user ID from execution context."""
        return self._execution_context.get("user_id", "")

    def get_channel(self) -> str:
        """Get channel from execution context."""
        return self._execution_context.get("channel", "")
