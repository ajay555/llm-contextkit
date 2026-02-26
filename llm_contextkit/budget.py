"""Token budget management for ContextKit.

Manages token allocation across context layers and ensures the assembled
context fits within model limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from llm_contextkit.exceptions import BudgetExceededError
from llm_contextkit.tokenizers.counting import TokenCounter, get_token_counter

logger = logging.getLogger("contextkit")


@dataclass
class LayerAllocation:
    """Allocation details for a single layer.

    Args:
        tokens: Maximum tokens allocated for this layer.
        priority: Priority for truncation ordering (higher = less likely to truncate).
        used: Actual tokens used by this layer after build.
    """

    tokens: int
    priority: int = 0
    used: int = 0


class TokenBudget:
    """Manages token allocation across context layers.

    Ensures the assembled context fits within model limits by tracking
    allocations and enforcing budgets during assembly.

    Args:
        total: Total token budget (model's context window or a subset of it).
        tokenizer: Tokenizer to use for counting. Options: "cl100k" (GPT-4/3.5),
            "o200k" (GPT-4o), "approximate" (word-based estimate), or a callable.
            Default: "cl100k".
        reserve_for_output: Tokens to reserve for model's response. Default: 1000.

    Raises:
        ValueError: If total or reserve_for_output is negative, or if
            reserve_for_output exceeds total.

    Examples:
        >>> budget = TokenBudget(total=4096, reserve_for_output=1000)
        >>> budget.allocate("system", 500, priority=10)
        >>> budget.allocate("history", 2500, priority=5)
        >>> budget.get_available()
        96
    """

    def __init__(
        self,
        total: int,
        tokenizer: str | Callable[[str], int] = "cl100k",
        reserve_for_output: int = 1000,
    ) -> None:
        if total < 0:
            raise ValueError("total must be non-negative")
        if reserve_for_output < 0:
            raise ValueError("reserve_for_output must be non-negative")
        if reserve_for_output > total:
            raise ValueError("reserve_for_output cannot exceed total")

        self._total = total
        self._reserve_for_output = reserve_for_output
        self._allocations: dict[str, LayerAllocation] = {}
        self._token_counter = get_token_counter(tokenizer)

    @property
    def total(self) -> int:
        """Return the total token budget."""
        return self._total

    @property
    def reserve_for_output(self) -> int:
        """Return the tokens reserved for model output."""
        return self._reserve_for_output

    @property
    def token_counter(self) -> TokenCounter:
        """Return the token counter instance."""
        return self._token_counter

    def allocate(self, layer_name: str, tokens: int, priority: int = 0) -> None:
        """Allocate a token budget to a named layer.

        Args:
            layer_name: Identifier for the context layer.
            tokens: Maximum tokens for this layer.
            priority: Higher priority layers get their full budget first.
                Lower priority layers are truncated first when over budget.
                Default: 0 (equal priority).

        Raises:
            ValueError: If tokens is negative.
            BudgetExceededError: If this allocation would cause total allocations
                plus reserve to exceed the total budget.
        """
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")

        # Calculate what the new total would be
        current_allocated = sum(a.tokens for a in self._allocations.values())
        if layer_name in self._allocations:
            current_allocated -= self._allocations[layer_name].tokens

        new_total_allocated = current_allocated + tokens
        available_for_allocation = self._total - self._reserve_for_output

        if new_total_allocated > available_for_allocation:
            raise BudgetExceededError(
                f"Allocation for '{layer_name}' ({tokens} tokens) would exceed budget. "
                f"Total allocated would be {new_total_allocated}, but only "
                f"{available_for_allocation} tokens available after reserving "
                f"{self._reserve_for_output} for output.",
                layer_name=layer_name,
                requested_tokens=tokens,
                available_tokens=available_for_allocation - current_allocated,
            )

        self._allocations[layer_name] = LayerAllocation(tokens=tokens, priority=priority)
        logger.debug(
            f"Allocated {tokens} tokens to layer '{layer_name}' with priority {priority}"
        )

    def get_allocation(self, layer_name: str) -> int:
        """Get the allocated budget for a layer.

        Args:
            layer_name: Identifier for the context layer.

        Returns:
            The number of tokens allocated to this layer, or 0 if not found.
        """
        if layer_name not in self._allocations:
            return 0
        return self._allocations[layer_name].tokens

    def get_priority(self, layer_name: str) -> int:
        """Get the priority for a layer.

        Args:
            layer_name: Identifier for the context layer.

        Returns:
            The priority of this layer, or 0 if not found.
        """
        if layer_name not in self._allocations:
            return 0
        return self._allocations[layer_name].priority

    def set_used(self, layer_name: str, tokens: int) -> None:
        """Record the actual tokens used by a layer after building.

        Args:
            layer_name: Identifier for the context layer.
            tokens: Actual tokens used by this layer.
        """
        if layer_name in self._allocations:
            self._allocations[layer_name].used = tokens

    def get_used(self, layer_name: str) -> int:
        """Get the actual tokens used by a layer.

        Args:
            layer_name: Identifier for the context layer.

        Returns:
            The number of tokens actually used by this layer.
        """
        if layer_name not in self._allocations:
            return 0
        return self._allocations[layer_name].used

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured tokenizer.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.
        """
        return self._token_counter.count(text)

    def get_available(self) -> int:
        """Get remaining unallocated tokens.

        Returns:
            The number of tokens not yet allocated to any layer.
        """
        total_allocated = sum(a.tokens for a in self._allocations.values())
        return self._total - self._reserve_for_output - total_allocated

    def get_total_allocated(self) -> int:
        """Get the total tokens allocated across all layers.

        Returns:
            The sum of all layer allocations.
        """
        return sum(a.tokens for a in self._allocations.values())

    def get_total_used(self) -> int:
        """Get the total tokens actually used across all layers.

        Returns:
            The sum of all layer usage.
        """
        return sum(a.used for a in self._allocations.values())

    def get_layers_by_priority(self) -> list[str]:
        """Get layer names sorted by priority (lowest first).

        Returns:
            List of layer names sorted by priority ascending.
        """
        return sorted(self._allocations.keys(), key=lambda k: self._allocations[k].priority)

    def summary(self) -> dict:
        """Return a summary of allocations vs actual usage.

        Returns:
            Dict with: total, reserved, allocated per layer,
            used per layer, remaining per layer, total_remaining.

        Examples:
            >>> budget = TokenBudget(total=4096, reserve_for_output=1000)
            >>> budget.allocate("system", 500, priority=10)
            >>> budget.summary()
            {
                'total': 4096,
                'reserved': 1000,
                'available_for_context': 3096,
                'total_allocated': 500,
                'total_used': 0,
                'unallocated': 2596,
                'layers': {
                    'system': {
                        'allocated': 500,
                        'used': 0,
                        'remaining': 500,
                        'priority': 10
                    }
                }
            }
        """
        layers_summary: dict[str, dict] = {}
        for name, alloc in self._allocations.items():
            layers_summary[name] = {
                "allocated": alloc.tokens,
                "used": alloc.used,
                "remaining": alloc.tokens - alloc.used,
                "priority": alloc.priority,
            }

        total_allocated = self.get_total_allocated()
        total_used = self.get_total_used()
        available_for_context = self._total - self._reserve_for_output

        return {
            "total": self._total,
            "reserved": self._reserve_for_output,
            "available_for_context": available_for_context,
            "total_allocated": total_allocated,
            "total_used": total_used,
            "unallocated": available_for_context - total_allocated,
            "layers": layers_summary,
        }
