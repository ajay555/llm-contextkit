"""System prompt layer for ContextKit.

Handles system instructions and optional few-shot examples.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from llm_contextkit.exceptions import BudgetExceededError
from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.tokenizers.counting import TokenCounter

logger = logging.getLogger("contextkit")


class FewShotExample(TypedDict):
    """A few-shot example with input and expected output."""

    input: str
    output: str


class SystemLayer(BaseLayer):
    """System prompt layer with static instructions and optional few-shot examples.

    The system layer typically has the highest priority (default 10) since
    system prompts are critical for model behavior and rarely should be truncated.

    Truncation behavior:
    - Few-shot examples are removed last-to-first before instructions are truncated
    - Instructions are never truncated; if they don't fit, BudgetExceededError is raised

    Args:
        instructions: The core system instructions.
        few_shot_examples: Optional list of {"input": str, "output": str} dicts.
        name: Layer name. Default: "system".
        priority: Truncation priority. Default: 10 (high priority).

    Examples:
        >>> layer = SystemLayer(
        ...     instructions="You are a helpful assistant.",
        ...     few_shot_examples=[
        ...         {"input": "Hi", "output": "Hello! How can I help you?"}
        ...     ]
        ... )
        >>> content = layer.build(token_counter)
    """

    def __init__(
        self,
        instructions: str,
        few_shot_examples: list[FewShotExample] | None = None,
        name: str = "system",
        priority: int = 10,
    ) -> None:
        self._instructions = instructions
        self._few_shot_examples = few_shot_examples or []
        self._included_examples: int = len(self._few_shot_examples)
        super().__init__(name=name, content=instructions, priority=priority)

    @property
    def instructions(self) -> str:
        """Return the system instructions."""
        return self._instructions

    @property
    def few_shot_examples(self) -> list[FewShotExample]:
        """Return the few-shot examples."""
        return self._few_shot_examples

    def build(self, token_counter: TokenCounter) -> str:
        """Build the system layer content.

        Assembles the instructions and few-shot examples into a formatted string.

        Args:
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The formatted system prompt content.
        """
        parts = [self._instructions]

        if self._few_shot_examples:
            examples_text = self._format_examples(self._few_shot_examples)
            parts.append(examples_text)

        self._built_content = "\n\n".join(parts)
        self._token_count = token_counter.count(self._built_content)
        self._included_examples = len(self._few_shot_examples)

        return self._built_content

    def truncate(self, max_tokens: int, token_counter: TokenCounter) -> str:
        """Truncate content to fit within max_tokens.

        Removes few-shot examples last-to-first. If instructions alone exceed
        max_tokens, raises BudgetExceededError.

        Args:
            max_tokens: Maximum number of tokens for this layer.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The truncated layer content.

        Raises:
            BudgetExceededError: If instructions alone exceed max_tokens.
        """
        # Check if instructions alone fit
        instructions_tokens = token_counter.count(self._instructions)
        if instructions_tokens > max_tokens:
            raise BudgetExceededError(
                f"System instructions ({instructions_tokens} tokens) exceed budget "
                f"({max_tokens} tokens). System instructions cannot be truncated.",
                layer_name=self._name,
                requested_tokens=instructions_tokens,
                available_tokens=max_tokens,
            )

        # If no examples, just return instructions
        if not self._few_shot_examples:
            self._built_content = self._instructions
            self._token_count = instructions_tokens
            return self._built_content

        # Try to include as many examples as possible
        included_examples: list[FewShotExample] = []
        for example in self._few_shot_examples:
            test_examples = included_examples + [example]
            test_content = self._format_with_examples(test_examples)
            if token_counter.count(test_content) <= max_tokens:
                included_examples.append(example)
            else:
                break

        # Build final content
        if included_examples:
            self._built_content = self._format_with_examples(included_examples)
        else:
            self._built_content = self._instructions

        self._token_count = token_counter.count(self._built_content)
        self._included_examples = len(included_examples)

        if len(included_examples) < len(self._few_shot_examples):
            self._truncated = True
            dropped = len(self._few_shot_examples) - len(included_examples)
            logger.info(
                f"SystemLayer '{self._name}': Dropped {dropped} few-shot examples "
                f"to fit within {max_tokens} token budget"
            )

        return self._built_content

    def _format_examples(self, examples: list[FewShotExample]) -> str:
        """Format few-shot examples as a string.

        Args:
            examples: List of example dicts.

        Returns:
            Formatted examples string.
        """
        formatted = []
        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"User: {example['input']}")
            formatted.append(f"Assistant: {example['output']}")
        return "\n".join(formatted)

    def _format_with_examples(self, examples: list[FewShotExample]) -> str:
        """Format instructions with examples.

        Args:
            examples: List of example dicts.

        Returns:
            Combined instructions and examples string.
        """
        if not examples:
            return self._instructions
        return f"{self._instructions}\n\n{self._format_examples(examples)}"

    def inspect(self) -> dict:
        """Return debug info about this layer.

        Returns:
            A dict containing layer metadata and debug information.
        """
        base_info = super().inspect()
        base_info.update(
            {
                "total_examples": len(self._few_shot_examples),
                "included_examples": self._included_examples,
                "dropped_examples": len(self._few_shot_examples) - self._included_examples,
            }
        )
        return base_info
