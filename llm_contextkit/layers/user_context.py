"""User context layer for ContextKit.

Handles user/session metadata that provides personalization context.
"""

import logging
from typing import Any

from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.tokenizers.counting import TokenCounter

logger = logging.getLogger("contextkit")


class UserContextLayer(BaseLayer):
    """User/session context layer for personalization data.

    Formats a dictionary of user metadata as structured key-value pairs
    for inclusion in the context.

    Truncation behavior:
    - Keys are removed by reverse insertion order (least important last-added
      fields are removed first)

    Args:
        context: Dict of user metadata (e.g., {"name": "Jane", "tier": "Enterprise"}).
        name: Layer name. Default: "user_context".
        priority: Truncation priority. Default: 7.

    Examples:
        >>> layer = UserContextLayer(
        ...     context={
        ...         "name": "Jane Smith",
        ...         "role": "Admin",
        ...         "account_tier": "Enterprise",
        ...     }
        ... )
        >>> content = layer.build(token_counter)
    """

    def __init__(
        self,
        context: dict[str, Any],
        name: str = "user_context",
        priority: int = 7,
    ) -> None:
        self._context_data = context.copy()  # Preserve original
        self._included_keys: list[str] = list(context.keys())
        super().__init__(name=name, content=context, priority=priority)

    @property
    def context_data(self) -> dict[str, Any]:
        """Return the user context data."""
        return self._context_data.copy()

    def build(self, token_counter: TokenCounter) -> str:
        """Build the user context layer content.

        Formats the context dict as structured key-value pairs.

        Args:
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The formatted user context content.
        """
        self._built_content = self._format_context(self._context_data)
        self._token_count = token_counter.count(self._built_content)
        self._included_keys = list(self._context_data.keys())

        return self._built_content

    def truncate(self, max_tokens: int, token_counter: TokenCounter) -> str:
        """Truncate content to fit within max_tokens.

        Removes keys in reverse insertion order (last-added first).

        Args:
            max_tokens: Maximum number of tokens for this layer.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The truncated layer content.
        """
        keys = list(self._context_data.keys())
        included_data: dict[str, Any] = {}

        # Try to include keys from first to last
        for key in keys:
            test_data = included_data.copy()
            test_data[key] = self._context_data[key]
            test_content = self._format_context(test_data)

            if token_counter.count(test_content) <= max_tokens:
                included_data[key] = self._context_data[key]
            else:
                break

        # Build final content
        if included_data:
            self._built_content = self._format_context(included_data)
        else:
            self._built_content = ""

        self._token_count = token_counter.count(self._built_content)
        self._included_keys = list(included_data.keys())

        if len(included_data) < len(self._context_data):
            self._truncated = True
            dropped = len(self._context_data) - len(included_data)
            logger.info(
                f"UserContextLayer '{self._name}': Dropped {dropped} keys "
                f"to fit within {max_tokens} token budget"
            )

        return self._built_content

    def _format_context(self, data: dict[str, Any]) -> str:
        """Format context data as key-value pairs.

        Args:
            data: The context data to format.

        Returns:
            Formatted string of key-value pairs.
        """
        if not data:
            return ""

        lines = []
        for key, value in data.items():
            # Convert key from snake_case to Title Case
            formatted_key = key.replace("_", " ").title()
            lines.append(f"- {formatted_key}: {value}")

        return "\n".join(lines)

    def inspect(self) -> dict:
        """Return debug info about this layer.

        Returns:
            A dict containing layer metadata and debug information.
        """
        base_info = super().inspect()
        base_info.update(
            {
                "total_keys": len(self._context_data),
                "included_keys": len(self._included_keys),
                "dropped_keys": len(self._context_data) - len(self._included_keys),
                "keys_included": self._included_keys,
            }
        )
        return base_info
