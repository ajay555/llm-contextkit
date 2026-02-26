"""Context formatting for ContextKit.

Provides formatters for structuring context layers with clear delimiters
and labels for better model comprehension.
"""

from __future__ import annotations

from llm_contextkit.layers.base import BaseLayer

# Layer name to human-readable section header mapping
DEFAULT_SECTION_LABELS: dict[str, str] = {
    "system": "System Instructions",
    "user_context": "Customer Profile",
    "retrieved_docs": "Relevant Documents",
    "history": "Conversation History",
    "tool_results": "Tool Results",
}


class DefaultFormatter:
    """Formats context layers with clear delimiters and labels.

    Default format uses markdown-style headers:

        ## System Instructions
        {system_content}

        ## Customer Profile
        {user_context}

        ## Relevant Documents
        [Source: policy-v4.pdf | Relevance: 0.92]
        {chunk_text}

        ## Conversation History
        User: ...
        Assistant: ...

    Args:
        section_delimiter: Delimiter for section headers. Default: "##".
        chunk_delimiter: Delimiter between chunks. Default: "---".
        include_token_hints: Whether to include token count hints. Default: False.

    Examples:
        >>> formatter = DefaultFormatter()
        >>> formatted = formatter.format_layer(system_layer)
    """

    def __init__(
        self,
        section_delimiter: str = "##",
        chunk_delimiter: str = "---",
        include_token_hints: bool = False,
    ) -> None:
        self._section_delimiter = section_delimiter
        self._chunk_delimiter = chunk_delimiter
        self._include_token_hints = include_token_hints
        self._section_labels = DEFAULT_SECTION_LABELS.copy()

    @property
    def section_delimiter(self) -> str:
        """Return the section delimiter."""
        return self._section_delimiter

    @property
    def chunk_delimiter(self) -> str:
        """Return the chunk delimiter."""
        return self._chunk_delimiter

    def set_section_label(self, layer_name: str, label: str) -> None:
        """Set a custom section label for a layer.

        Args:
            layer_name: The layer name to set the label for.
            label: The human-readable label for the section.
        """
        self._section_labels[layer_name] = label

    def get_section_label(self, layer_name: str) -> str:
        """Get the section label for a layer.

        Args:
            layer_name: The layer name to get the label for.

        Returns:
            The human-readable label, or the layer name if no label is set.
        """
        return self._section_labels.get(layer_name, layer_name.replace("_", " ").title())

    def format_section_header(self, layer_name: str, token_count: int | None = None) -> str:
        """Format a section header for a layer.

        Args:
            layer_name: The layer name.
            token_count: Optional token count to include in the header.

        Returns:
            The formatted section header.
        """
        label = self.get_section_label(layer_name)
        header = f"{self._section_delimiter} {label}"

        if self._include_token_hints and token_count is not None:
            header += f" [{token_count} tokens]"

        return header

    def format_layer(self, layer: BaseLayer) -> str:
        """Format a single layer with appropriate delimiters.

        This method formats the layer's built content with a section header.
        The layer must be built before calling this method.

        Args:
            layer: The layer to format.

        Returns:
            The formatted layer content with header.

        Raises:
            ValueError: If the layer has not been built.
        """
        if layer.built_content is None:
            raise ValueError(f"Layer '{layer.name}' has not been built yet")

        token_count = layer.get_token_count() if self._include_token_hints else None
        header = self.format_section_header(layer.name, token_count)

        return f"{header}\n{layer.built_content}"

    def format_all(self, layers: dict[str, BaseLayer]) -> str:
        """Format all layers into a single context string.

        Layers are formatted in the order they appear in the dict.
        Each layer is separated by a blank line.

        Args:
            layers: Dict mapping layer names to layer instances.

        Returns:
            The combined formatted context string.
        """
        formatted_parts = []

        for layer in layers.values():
            if layer.built_content:
                formatted_parts.append(self.format_layer(layer))

        return "\n\n".join(formatted_parts)

    def format_message(self, role: str, content: str) -> str:
        """Format a single message.

        Args:
            role: The message role (e.g., "user", "assistant").
            content: The message content.

        Returns:
            The formatted message.
        """
        role_label = role.capitalize()
        return f"{role_label}: {content}"

    def format_chunk(
        self,
        text: str,
        source: str | None = None,
        score: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Format a retrieved chunk with metadata.

        Args:
            text: The chunk text.
            source: Optional source identifier.
            score: Optional relevance score.
            metadata: Optional additional metadata.

        Returns:
            The formatted chunk.
        """
        parts = []

        # Build metadata line
        meta_parts = []
        if source:
            meta_parts.append(f"Source: {source}")
        if score is not None:
            meta_parts.append(f"Relevance: {score:.2f}")

        if meta_parts:
            parts.append(f"[{' | '.join(meta_parts)}]")

        parts.append(text)

        return "\n".join(parts)
