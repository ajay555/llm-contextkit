"""Context assembler for ContextKit.

The ContextAssembler is the core orchestrator that brings together all layers
and manages the context assembly process.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_contextkit.budget import TokenBudget
from llm_contextkit.exceptions import BudgetExceededError, BuildError
from llm_contextkit.formatting.formatter import DefaultFormatter
from llm_contextkit.layers.base import BaseLayer

logger = logging.getLogger("contextkit")


class ContextAssembler:
    """Assembles context layers into a final payload respecting token budgets.

    The ContextAssembler is the main entry point for building context. It manages
    layers, enforces token budgets, handles truncation, and provides multiple
    output formats for different LLM APIs.

    Args:
        budget: TokenBudget instance for managing token allocations.
        formatter: Optional Formatter instance for custom formatting.
            Default: DefaultFormatter().

    Examples:
        >>> budget = TokenBudget(total=4096, reserve_for_output=1000)
        >>> budget.allocate("system", 500, priority=10)
        >>> assembler = ContextAssembler(budget=budget)
        >>> assembler.add_layer(SystemLayer(instructions="You are helpful."))
        >>> result = assembler.build()
    """

    def __init__(
        self,
        budget: TokenBudget,
        formatter: DefaultFormatter | None = None,
    ) -> None:
        self._budget = budget
        self._formatter = formatter or DefaultFormatter()
        self._layers: dict[str, BaseLayer] = {}
        self._build_result: dict | None = None
        self._build_metadata: dict | None = None
        self._warnings: list[str] = []

    @property
    def budget(self) -> TokenBudget:
        """Return the token budget."""
        return self._budget

    @property
    def formatter(self) -> DefaultFormatter:
        """Return the formatter."""
        return self._formatter

    @property
    def layers(self) -> dict[str, BaseLayer]:
        """Return the layers dict."""
        return self._layers.copy()

    def add_layer(self, layer: BaseLayer) -> ContextAssembler:
        """Add a context layer.

        Returns self for chaining.

        Args:
            layer: The layer to add.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If a layer with the same name already exists.

        Examples:
            >>> assembler.add_layer(system_layer).add_layer(history_layer)
        """
        if layer.name in self._layers:
            raise ValueError(f"Layer '{layer.name}' already exists")

        self._layers[layer.name] = layer
        logger.debug(f"Added layer '{layer.name}' with priority {layer.priority}")
        return self

    def remove_layer(self, name: str) -> ContextAssembler:
        """Remove a layer by name.

        Args:
            name: The name of the layer to remove.

        Returns:
            Self for method chaining.

        Raises:
            KeyError: If no layer with the given name exists.
        """
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found")

        del self._layers[name]
        logger.debug(f"Removed layer '{name}'")
        return self

    def get_layer(self, name: str) -> BaseLayer | None:
        """Get a layer by name.

        Args:
            name: The name of the layer.

        Returns:
            The layer, or None if not found.
        """
        return self._layers.get(name)

    def build(self) -> dict:
        """Assemble all layers into the final context payload.

        Build process:
        1. Build all layers
        2. Check total tokens against budget
        3. If over budget, truncate layers in priority order (lowest first)
        4. Format using the configured formatter
        5. Store build result for inspection

        Returns:
            A dict with:
            {
                "system": str,          # System message content
                "messages": list,       # Conversation messages (history + current)
                "metadata": dict        # Build metadata (token counts, truncations, etc.)
            }

        Raises:
            BuildError: If context assembly fails.
            BudgetExceededError: If context cannot fit even after truncation.
        """
        start_time = time.perf_counter()
        self._warnings = []

        try:
            # Build all layers
            self._build_all_layers()

            # Check and handle budget
            self._enforce_budget()

            # Extract system content
            system_content = self._extract_system_content()

            # Extract messages (if we have a history layer)
            messages = self._extract_messages()

            # Calculate build time
            build_time_ms = (time.perf_counter() - start_time) * 1000

            # Record usage in budget
            for layer in self._layers.values():
                self._budget.set_used(layer.name, layer.get_token_count())

            # Build metadata
            total_tokens = sum(layer.get_token_count() for layer in self._layers.values())
            self._build_metadata = {
                "total_tokens": total_tokens,
                "budget_total": self._budget.total,
                "budget_remaining": self._budget.total
                - self._budget.reserve_for_output
                - total_tokens,
                "build_time_ms": build_time_ms,
                "warnings": self._warnings,
                "layers": {
                    name: {
                        "tokens": layer.get_token_count(),
                        "allocated": self._budget.get_allocation(name),
                        "truncated": layer.truncated,
                    }
                    for name, layer in self._layers.items()
                },
            }

            self._build_result = {
                "system": system_content,
                "messages": messages,
                "metadata": self._build_metadata,
            }

            logger.debug(
                f"Build complete: {total_tokens} tokens in {build_time_ms:.2f}ms"
            )
            return self._build_result

        except BudgetExceededError:
            raise
        except Exception as e:
            raise BuildError(f"Context assembly failed: {e}", phase="build") from e

    def _build_all_layers(self) -> None:
        """Build all layers with the token counter."""
        token_counter = self._budget.token_counter

        for layer in self._layers.values():
            layer.build(token_counter)
            logger.debug(
                f"Built layer '{layer.name}': {layer.get_token_count()} tokens"
            )

    def _enforce_budget(self) -> None:
        """Enforce token budget by truncating layers if necessary."""
        token_counter = self._budget.token_counter
        available = self._budget.total - self._budget.reserve_for_output

        # Calculate current total
        total_tokens = sum(layer.get_token_count() for layer in self._layers.values())

        if total_tokens <= available:
            return  # We're under budget

        logger.info(
            f"Over budget: {total_tokens} tokens used, {available} available. "
            f"Truncating layers..."
        )

        # Get layers sorted by priority (lowest first = truncate first)
        layers_by_priority = sorted(
            self._layers.values(), key=lambda layer: layer.priority
        )

        for layer in layers_by_priority:
            if total_tokens <= available:
                break

            # Calculate how much this layer needs to shrink
            layer_allocation = self._budget.get_allocation(layer.name)
            if layer_allocation == 0:
                # No explicit allocation, use fair share of remaining budget
                layer_allocation = layer.get_token_count()

            # Calculate max tokens for this layer
            excess = total_tokens - available
            layer_current = layer.get_token_count()
            max_tokens = max(0, layer_current - excess)

            # Ensure we don't give more than the allocation
            max_tokens = min(max_tokens, layer_allocation)

            if max_tokens < layer_current:
                old_tokens = layer_current

                # Truncate if layer priority < 10 (high priority layers can't be truncated)
                if layer.priority >= 10:
                    raise BudgetExceededError(
                        f"Cannot fit context within budget. High-priority layer "
                        f"'{layer.name}' ({layer_current} tokens) exceeds remaining "
                        f"budget ({available - (total_tokens - layer_current)} tokens).",
                        layer_name=layer.name,
                        requested_tokens=layer_current,
                        available_tokens=available - (total_tokens - layer_current),
                    )

                layer.truncate(max_tokens, token_counter)
                new_tokens = layer.get_token_count()
                total_tokens = total_tokens - old_tokens + new_tokens

                warning = (
                    f"Layer '{layer.name}' truncated from {old_tokens} "
                    f"to {new_tokens} tokens"
                )
                self._warnings.append(warning)
                logger.info(warning)

        # Final check
        if total_tokens > available:
            raise BudgetExceededError(
                f"Cannot fit context within budget after truncation. "
                f"Total: {total_tokens} tokens, Available: {available} tokens.",
                requested_tokens=total_tokens,
                available_tokens=available,
            )

    def _extract_system_content(self) -> str:
        """Extract system content from layers."""
        # Look for system layer
        system_layer = self._layers.get("system")
        if system_layer and system_layer.built_content:
            return system_layer.built_content

        # Try to find any layer with "system" in the name
        for name, layer in self._layers.items():
            if "system" in name.lower() and layer.built_content:
                return layer.built_content

        return ""

    def _extract_messages(self) -> list[dict[str, str]]:
        """Extract messages from history layer."""
        # Look for history layer
        history_layer = self._layers.get("history")
        if history_layer and hasattr(history_layer, "processed_messages"):
            # Use processed_messages for HistoryLayer
            messages: list[dict[str, str]] = history_layer.processed_messages
            return messages
        if history_layer and hasattr(history_layer, "content"):
            content = history_layer.content
            if isinstance(content, list):
                return content

        return []

    def _build_full_system_content(self) -> str:
        """Build complete system content including all context layers.

        Combines system instructions, user context, retrieved docs, and
        tool results into a comprehensive system message.

        Returns:
            The complete system content string.
        """
        parts = []

        # 1. System instructions (highest priority)
        system_layer = self._layers.get("system")
        if system_layer and system_layer.built_content:
            parts.append(system_layer.built_content)

        # 2. User context
        user_context_layer = self._layers.get("user_context")
        if user_context_layer and user_context_layer.built_content:
            parts.append(
                f"{self._formatter.format_section_header('user_context')}\n"
                f"{user_context_layer.built_content}"
            )

        # 3. Retrieved documents
        retrieved_layer = self._layers.get("retrieved_docs")
        if retrieved_layer and retrieved_layer.built_content:
            parts.append(
                f"{self._formatter.format_section_header('retrieved_docs')}\n"
                f"{retrieved_layer.built_content}"
            )

        # 4. Tool results (for agentic workflows)
        tool_results_layer = self._layers.get("tool_results")
        if tool_results_layer and tool_results_layer.built_content:
            parts.append(
                f"{self._formatter.format_section_header('tool_results')}\n"
                f"{tool_results_layer.built_content}"
            )

        return "\n\n".join(parts)

    def build_for_openai(self) -> list[dict[str, str]]:
        """Build and return in OpenAI messages format.

        Combines all layers into the OpenAI chat format. System instructions,
        user context, retrieved documents, and tool results are combined into
        the system message. History messages are included as conversation turns.

        Returns:
            A list of message dicts in OpenAI format:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]

        Examples:
            >>> messages = assembler.build_for_openai()
            >>> response = openai.chat.completions.create(
            ...     model="gpt-4",
            ...     messages=messages
            ... )
        """
        self.build()  # Ensure build is done

        messages: list[dict[str, str]] = []

        # Build comprehensive system content
        system_content = self._build_full_system_content()
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Add conversation history
        history_messages = self._extract_messages()
        for msg in history_messages:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        return messages

    def build_for_anthropic(self) -> dict[str, Any]:
        """Build and return in Anthropic API format.

        Combines all layers into the Anthropic chat format. System instructions,
        user context, retrieved documents, and tool results are combined into
        the system parameter. History messages are included in the messages array.

        Note: Anthropic requires messages to start with a user message and
        alternate between user and assistant roles.

        Returns:
            A dict in Anthropic format:
            {
                "system": "...",
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ]
            }

        Examples:
            >>> payload = assembler.build_for_anthropic()
            >>> response = anthropic.messages.create(
            ...     model="claude-3-opus-20240229",
            ...     system=payload["system"],
            ...     messages=payload["messages"]
            ... )
        """
        self.build()  # Ensure build is done

        # Build comprehensive system content
        system_content = self._build_full_system_content()

        # Extract and validate messages for Anthropic format
        history_messages = self._extract_messages()
        messages: list[dict[str, str]] = []

        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic only accepts "user" and "assistant" roles
            if role not in ("user", "assistant"):
                role = "user"  # Default to user for unknown roles

            messages.append({"role": role, "content": content})

        return {
            "system": system_content,
            "messages": messages,
        }

    def inspect(self) -> dict:
        """Return detailed debug information about the last build.

        Returns:
            A dict with build details:
            {
                "total_tokens": int,
                "budget_total": int,
                "budget_remaining": int,
                "layers": {
                    "system": {"tokens": int, "allocated": int, "truncated": bool},
                    ...
                },
                "warnings": ["..."],
                "build_time_ms": float
            }

        Raises:
            ValueError: If build() has not been called yet.
        """
        if self._build_metadata is None:
            raise ValueError("No build result available. Call build() first.")

        return self._build_metadata.copy()

    def inspect_pretty(self) -> str:
        """Return a human-readable formatted string of inspect() output.

        Returns:
            A formatted string showing the build summary.

        Raises:
            ValueError: If build() has not been called yet.
        """
        info = self.inspect()

        lines = [
            "┌─────────────────────────────────────────┐",
            "│ Context Build Summary                   │",
            "├─────────────┬────────┬───────┬──────────┤",
            "│ Layer       │ Tokens │ Alloc │ Truncated│",
            "├─────────────┼────────┼───────┼──────────┤",
        ]

        for name, layer_info in info["layers"].items():
            truncated = "Yes" if layer_info["truncated"] else "No"
            lines.append(
                f"│ {name:<11} │ {layer_info['tokens']:>6,} │ {layer_info['allocated']:>5,} │ {truncated:<8} │"
            )

        lines.extend(
            [
                "├─────────────┼────────┼───────┼──────────┤",
                f"│ Total       │ {info['total_tokens']:>6,} │       │          │",
                f"│ Reserved    │ {self._budget.reserve_for_output:>6,} │       │          │",
                f"│ Remaining   │ {info['budget_remaining']:>6,} │       │          │",
                "└─────────────┴────────┴───────┴──────────┘",
            ]
        )

        if info["warnings"]:
            lines.append("")
            lines.append("Warnings:")
            for warning in info["warnings"]:
                lines.append(f"  - {warning}")

        lines.append("")
        lines.append(f"Build time: {info['build_time_ms']:.2f}ms")

        return "\n".join(lines)
