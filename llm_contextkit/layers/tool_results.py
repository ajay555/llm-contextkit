"""Tool results layer for ContextKit.

Handles tool/API call results for agentic workflows.
"""

import logging
from typing import Any, Literal, TypedDict

from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.tokenizers.counting import TokenCounter

logger = logging.getLogger("contextkit")


class ToolResult(TypedDict, total=False):
    """A tool/API call result."""

    tool_name: str  # Required
    output: str  # Required
    status: Literal["success", "error"]  # Required
    input: dict[str, Any]  # Optional


class ToolResultsLayer(BaseLayer):
    """Tool/API call results layer for agentic workflows.

    Formats tool results for inclusion in context, allowing the model to
    reason about previous tool calls and their outputs.

    Truncation behavior:
    - Older tool results are summarized (keep tool_name + status, drop full output)
    - Most recent tool result is never truncated

    Args:
        results: List of dicts with: tool_name, input, output, status.
        include_inputs: Whether to include tool inputs. Default: False.
        name: Layer name. Default: "tool_results".
        priority: Truncation priority. Default: 8 (high - agent needs these).

    Examples:
        >>> layer = ToolResultsLayer(
        ...     results=[
        ...         {"tool_name": "web_search", "input": {"query": "weather"},
        ...          "output": "Sunny, 72°F", "status": "success"},
        ...     ],
        ...     include_inputs=True,
        ... )
        >>> content = layer.build(token_counter)
    """

    def __init__(
        self,
        results: list[ToolResult],
        include_inputs: bool = False,
        name: str = "tool_results",
        priority: int = 8,
    ) -> None:
        self._results = [result.copy() for result in results]  # Deep copy
        self._include_inputs = include_inputs
        self._full_results: int = 0
        self._summarized_results: int = 0

        super().__init__(name=name, content=results, priority=priority)

    @property
    def results(self) -> list[ToolResult]:
        """Return the original results."""
        return [result.copy() for result in self._results]

    @property
    def include_inputs(self) -> bool:
        """Return whether inputs are included."""
        return self._include_inputs

    def build(self, token_counter: TokenCounter) -> str:
        """Build the tool results layer content.

        Args:
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The formatted tool results content.
        """
        self._built_content = self._format_results(self._results, summarize_count=0)
        self._token_count = token_counter.count(self._built_content)
        self._full_results = len(self._results)
        self._summarized_results = 0

        return self._built_content

    def truncate(self, max_tokens: int, token_counter: TokenCounter) -> str:
        """Truncate content to fit within max_tokens.

        Summarizes older tool results while keeping the most recent one intact.

        Args:
            max_tokens: Maximum number of tokens for this layer.
            token_counter: TokenCounter instance for counting tokens.

        Returns:
            The truncated layer content.
        """
        if not self._results:
            self._built_content = ""
            self._token_count = 0
            return ""

        # Try full content first
        content = self._format_results(self._results, summarize_count=0)
        if token_counter.count(content) <= max_tokens:
            self._built_content = content
            self._token_count = token_counter.count(content)
            self._full_results = len(self._results)
            self._summarized_results = 0
            return content

        # Progressively summarize older results
        for summarize_count in range(1, len(self._results)):
            content = self._format_results(self._results, summarize_count=summarize_count)
            if token_counter.count(content) <= max_tokens:
                self._truncated = True
                self._built_content = content
                self._token_count = token_counter.count(content)
                self._full_results = len(self._results) - summarize_count
                self._summarized_results = summarize_count
                logger.info(
                    f"ToolResultsLayer '{self._name}': Summarized {summarize_count} "
                    f"older results to fit within {max_tokens} token budget"
                )
                return content

        # If even all summarized doesn't fit, just keep the most recent
        self._truncated = True
        content = self._format_results(
            self._results[-1:], summarize_count=0
        )

        # If still too large, return minimal info
        if token_counter.count(content) > max_tokens:
            last_result = self._results[-1]
            content = (
                f"[Most recent tool call: {last_result['tool_name']} - "
                f"{last_result.get('status', 'unknown')}]"
            )

        self._built_content = content
        self._token_count = token_counter.count(content)
        self._full_results = 1
        self._summarized_results = len(self._results) - 1

        return content

    def _format_results(
        self, results: list[ToolResult], summarize_count: int = 0
    ) -> str:
        """Format tool results.

        Args:
            results: List of tool result dicts.
            summarize_count: Number of oldest results to summarize.

        Returns:
            Formatted tool results string.
        """
        if not results:
            return ""

        formatted = []

        for i, result in enumerate(results):
            if i < summarize_count:
                # Summarized format
                formatted.append(self._format_summary(result))
            else:
                # Full format
                formatted.append(self._format_full(result))

        return "\n\n".join(formatted)

    def _format_summary(self, result: ToolResult) -> str:
        """Format a summarized tool result.

        Args:
            result: Tool result dict.

        Returns:
            Summarized result string.
        """
        status = result.get("status", "unknown")
        status_icon = "✓" if status == "success" else "✗"
        return f"[{result['tool_name']}: {status_icon} {status}]"

    def _format_full(self, result: ToolResult) -> str:
        """Format a full tool result.

        Args:
            result: Tool result dict.

        Returns:
            Full result string.
        """
        lines = []

        # Tool name and status
        status = result.get("status", "unknown")
        status_label = "Success" if status == "success" else "Error"
        lines.append(f"Tool: {result['tool_name']} ({status_label})")

        # Input (if enabled and present)
        if self._include_inputs and "input" in result:
            input_str = self._format_input(result["input"])
            lines.append(f"Input: {input_str}")

        # Output
        lines.append(f"Output: {result['output']}")

        return "\n".join(lines)

    def _format_input(self, input_data: dict[str, Any]) -> str:
        """Format tool input for display.

        Args:
            input_data: Input dictionary.

        Returns:
            Formatted input string.
        """
        if not input_data:
            return "{}"

        parts = []
        for key, value in input_data.items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:47] + "..."
            parts.append(f"{key}={value!r}")

        return "{" + ", ".join(parts) + "}"

    def inspect(self) -> dict:
        """Return debug info about this layer.

        Returns:
            A dict containing layer metadata and debug information.
        """
        base_info = super().inspect()
        base_info.update(
            {
                "total_results": len(self._results),
                "full_results": self._full_results,
                "summarized_results": self._summarized_results,
                "include_inputs": self._include_inputs,
            }
        )
        return base_info
