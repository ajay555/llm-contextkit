"""Context inspector and debugger for ContextKit.

Provides tools for analyzing, debugging, and comparing context payloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from llm_contextkit.tokenizers.counting import TokenCounter, get_token_counter

logger = logging.getLogger("contextkit")


# Approximate pricing per 1M tokens (as of 2024)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


@dataclass
class MessageAnalysis:
    """Analysis of a single message."""

    index: int
    role: str
    token_count: int
    char_count: int
    content_preview: str


@dataclass
class InspectionReport:
    """Report from analyzing a context payload."""

    total_tokens: int
    message_count: int
    messages: list[MessageAnalysis]
    token_distribution: dict[str, int]
    token_percentages: dict[str, float]
    warnings: list[str]
    estimated_costs: dict[str, float]

    def pretty(self) -> str:
        """Return a human-readable formatted report.

        Returns:
            Formatted string representation of the report.
        """
        lines = [
            "┌─────────────────────────────────────────────────┐",
            "│           Context Inspection Report             │",
            "└─────────────────────────────────────────────────┘",
            "",
            f"Total Tokens: {self.total_tokens:,}",
            f"Message Count: {self.message_count}",
            "",
            "Token Distribution:",
        ]

        for role, count in self.token_distribution.items():
            pct = self.token_percentages.get(role, 0)
            bar_len = int(pct / 5)  # 20 chars max
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {role:12} {bar} {count:>6,} ({pct:>5.1f}%)")

        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        lines.extend(["", "Estimated Input Costs (per request):"])
        for model, cost in self.estimated_costs.items():
            lines.append(f"  {model:16} ${cost:.6f}")

        lines.extend(["", "Messages:"])
        for msg in self.messages[:10]:  # Show first 10
            preview = msg.content_preview[:40] + "..." if len(msg.content_preview) > 40 else msg.content_preview
            lines.append(f"  [{msg.index}] {msg.role:10} {msg.token_count:>5} tokens: {preview}")

        if len(self.messages) > 10:
            lines.append(f"  ... and {len(self.messages) - 10} more messages")

        return "\n".join(lines)


@dataclass
class InspectionDiff:
    """Diff between two context payloads."""

    before_tokens: int
    after_tokens: int
    token_delta: int
    messages_added: list[MessageAnalysis]
    messages_removed: list[MessageAnalysis]
    messages_changed: list[tuple[MessageAnalysis, MessageAnalysis]]

    def pretty(self) -> str:
        """Return a human-readable formatted diff.

        Returns:
            Formatted string representation of the diff.
        """
        delta_sign = "+" if self.token_delta >= 0 else ""
        lines = [
            "┌─────────────────────────────────────────────────┐",
            "│             Context Diff Report                 │",
            "└─────────────────────────────────────────────────┘",
            "",
            f"Before: {self.before_tokens:,} tokens",
            f"After:  {self.after_tokens:,} tokens",
            f"Delta:  {delta_sign}{self.token_delta:,} tokens",
            "",
        ]

        if self.messages_added:
            lines.append(f"Messages Added ({len(self.messages_added)}):")
            for msg in self.messages_added[:5]:
                preview = msg.content_preview[:30] + "..." if len(msg.content_preview) > 30 else msg.content_preview
                lines.append(f"  + [{msg.role}] {preview}")
            if len(self.messages_added) > 5:
                lines.append(f"  ... and {len(self.messages_added) - 5} more")
            lines.append("")

        if self.messages_removed:
            lines.append(f"Messages Removed ({len(self.messages_removed)}):")
            for msg in self.messages_removed[:5]:
                preview = msg.content_preview[:30] + "..." if len(msg.content_preview) > 30 else msg.content_preview
                lines.append(f"  - [{msg.role}] {preview}")
            if len(self.messages_removed) > 5:
                lines.append(f"  ... and {len(self.messages_removed) - 5} more")
            lines.append("")

        if self.messages_changed:
            lines.append(f"Messages Changed ({len(self.messages_changed)}):")
            for before, after in self.messages_changed[:5]:
                delta = after.token_count - before.token_count
                delta_str = f"+{delta}" if delta >= 0 else str(delta)
                lines.append(f"  ~ [{before.role}] {before.token_count} → {after.token_count} ({delta_str})")
            if len(self.messages_changed) > 5:
                lines.append(f"  ... and {len(self.messages_changed) - 5} more")

        return "\n".join(lines)


@dataclass
class BuildTrace:
    """Trace of a context assembly build process."""

    layers_built: list[dict[str, Any]]
    truncations: list[dict[str, Any]]
    final_tokens: int
    budget_total: int
    budget_used: int
    build_time_ms: float
    warnings: list[str]

    def pretty(self) -> str:
        """Return a human-readable formatted trace.

        Returns:
            Formatted string representation of the trace.
        """
        lines = [
            "┌─────────────────────────────────────────────────┐",
            "│              Build Trace Report                 │",
            "└─────────────────────────────────────────────────┘",
            "",
            f"Build Time: {self.build_time_ms:.2f}ms",
            f"Budget: {self.budget_used:,} / {self.budget_total:,} tokens",
            "",
            "Layers Built:",
        ]

        for layer in self.layers_built:
            status = "✓" if not layer.get("truncated") else "⚠ truncated"
            lines.append(
                f"  {layer['name']:15} {layer['tokens']:>6,} tokens [{status}]"
            )

        if self.truncations:
            lines.extend(["", "Truncation Events:"])
            for trunc in self.truncations:
                lines.append(
                    f"  {trunc['layer']:15} {trunc['before']:>6,} → {trunc['after']:>6,} tokens"
                )

        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)


class ContextInspector:
    """Standalone inspector for debugging context issues.

    Can be used independently of ContextAssembler to analyze any
    messages payload.

    Args:
        tokenizer: Tokenizer to use for counting. Options: "cl100k" (GPT-4/3.5),
            "o200k" (GPT-4o), "approximate". Default: "cl100k".

    Examples:
        >>> inspector = ContextInspector(tokenizer="cl100k")
        >>> report = inspector.analyze(messages)
        >>> print(report.pretty())
    """

    def __init__(self, tokenizer: str = "cl100k") -> None:
        self._token_counter = get_token_counter(tokenizer)

    @property
    def token_counter(self) -> TokenCounter:
        """Return the token counter."""
        return self._token_counter

    def analyze(self, messages: list[dict[str, str]]) -> InspectionReport:
        """Analyze an existing messages payload.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            InspectionReport with detailed analysis.
        """
        message_analyses: list[MessageAnalysis] = []
        token_distribution: dict[str, int] = {}
        total_tokens = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            token_count = self._token_counter.count(content)
            char_count = len(content)

            analysis = MessageAnalysis(
                index=i,
                role=role,
                token_count=token_count,
                char_count=char_count,
                content_preview=content[:100],
            )
            message_analyses.append(analysis)

            total_tokens += token_count
            token_distribution[role] = token_distribution.get(role, 0) + token_count

        # Calculate percentages
        token_percentages: dict[str, float] = {}
        if total_tokens > 0:
            for role, count in token_distribution.items():
                token_percentages[role] = (count / total_tokens) * 100

        # Generate warnings
        warnings = self._generate_warnings(
            total_tokens, token_distribution, token_percentages, messages
        )

        # Estimate costs
        estimated_costs = self._estimate_costs(total_tokens)

        return InspectionReport(
            total_tokens=total_tokens,
            message_count=len(messages),
            messages=message_analyses,
            token_distribution=token_distribution,
            token_percentages=token_percentages,
            warnings=warnings,
            estimated_costs=estimated_costs,
        )

    def diff(
        self, before: list[dict[str, str]], after: list[dict[str, str]]
    ) -> InspectionDiff:
        """Compare two context payloads and show what changed.

        Args:
            before: The original messages list.
            after: The modified messages list.

        Returns:
            InspectionDiff showing the differences.
        """
        before_analysis = self.analyze(before)
        after_analysis = self.analyze(after)

        # Create content-based lookup for comparison
        before_contents = {
            (msg["role"], msg.get("content", "")): i for i, msg in enumerate(before)
        }
        after_contents = {
            (msg["role"], msg.get("content", "")): i for i, msg in enumerate(after)
        }

        # Find added messages
        messages_added = []
        for key, idx in after_contents.items():
            if key not in before_contents:
                messages_added.append(after_analysis.messages[idx])

        # Find removed messages
        messages_removed = []
        for key, idx in before_contents.items():
            if key not in after_contents:
                messages_removed.append(before_analysis.messages[idx])

        # Find changed messages (same index, different content)
        messages_changed = []
        for i in range(min(len(before), len(after))):
            if before[i].get("content") != after[i].get("content"):
                if before[i].get("role") == after[i].get("role"):
                    messages_changed.append(
                        (before_analysis.messages[i], after_analysis.messages[i])
                    )

        return InspectionDiff(
            before_tokens=before_analysis.total_tokens,
            after_tokens=after_analysis.total_tokens,
            token_delta=after_analysis.total_tokens - before_analysis.total_tokens,
            messages_added=messages_added,
            messages_removed=messages_removed,
            messages_changed=messages_changed,
        )

    def trace(self, assembler: Any) -> BuildTrace:
        """Attach to an assembler and record the build trace.

        This method builds the context and captures detailed information
        about what was included, excluded, truncated, and why.

        Args:
            assembler: A ContextAssembler instance.

        Returns:
            BuildTrace with detailed build information.
        """
        # Trigger a build if not already done
        try:
            assembler.build()
        except Exception:
            pass  # May already be built or have issues

        # Get inspection data
        try:
            inspect_data = assembler.inspect()
        except ValueError:
            # No build result available
            return BuildTrace(
                layers_built=[],
                truncations=[],
                final_tokens=0,
                budget_total=0,
                budget_used=0,
                build_time_ms=0,
                warnings=["No build result available"],
            )

        # Build layer info
        layers_built = []
        truncations = []

        for name, layer_info in inspect_data.get("layers", {}).items():
            layers_built.append({
                "name": name,
                "tokens": layer_info.get("tokens", 0),
                "allocated": layer_info.get("allocated", 0),
                "truncated": layer_info.get("truncated", False),
            })

            if layer_info.get("truncated"):
                truncations.append({
                    "layer": name,
                    "before": layer_info.get("allocated", 0),
                    "after": layer_info.get("tokens", 0),
                })

        return BuildTrace(
            layers_built=layers_built,
            truncations=truncations,
            final_tokens=inspect_data.get("total_tokens", 0),
            budget_total=inspect_data.get("budget_total", 0),
            budget_used=inspect_data.get("total_tokens", 0),
            build_time_ms=inspect_data.get("build_time_ms", 0),
            warnings=inspect_data.get("warnings", []),
        )

    def _generate_warnings(
        self,
        total_tokens: int,
        distribution: dict[str, int],
        percentages: dict[str, float],
        messages: list[dict[str, str]],
    ) -> list[str]:
        """Generate warnings based on analysis.

        Args:
            total_tokens: Total token count.
            distribution: Token distribution by role.
            percentages: Token percentages by role.
            messages: Original messages.

        Returns:
            List of warning strings.
        """
        warnings = []

        # Check if system prompt is too large
        system_pct = percentages.get("system", 0)
        if system_pct > 40:
            warnings.append(
                f"System prompt uses {system_pct:.1f}% of context window - "
                "consider condensing"
            )

        # Check for very large individual messages
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            msg_tokens = self._token_counter.count(content)
            if total_tokens > 0 and msg_tokens / total_tokens > 0.3:
                warnings.append(
                    f"Message {i} ({msg.get('role', 'unknown')}) uses "
                    f"{msg_tokens / total_tokens * 100:.1f}% of total tokens"
                )

        # Check for potential context window issues
        if total_tokens > 100000:
            warnings.append(
                f"Very large context ({total_tokens:,} tokens) - "
                "ensure model supports this context length"
            )
        elif total_tokens > 32000:
            warnings.append(
                f"Large context ({total_tokens:,} tokens) - "
                "may not fit in all model context windows"
            )

        # Check for empty messages
        empty_count = sum(1 for m in messages if not m.get("content", "").strip())
        if empty_count > 0:
            warnings.append(f"{empty_count} message(s) have empty content")

        return warnings

    def _estimate_costs(self, tokens: int) -> dict[str, float]:
        """Estimate costs for different models.

        Args:
            tokens: Token count.

        Returns:
            Dict mapping model names to estimated input costs.
        """
        costs = {}
        for model, pricing in MODEL_PRICING.items():
            # Cost per token = price per million / 1,000,000
            cost = tokens * (pricing["input"] / 1_000_000)
            costs[model] = cost
        return costs
