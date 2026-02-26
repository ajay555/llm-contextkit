"""Pre-built formatting templates for ContextKit.

Provides XMLFormatter and MinimalFormatter as alternatives to DefaultFormatter.
"""

from __future__ import annotations

from llm_contextkit.formatting.formatter import DefaultFormatter
from llm_contextkit.layers.base import BaseLayer


class XMLFormatter(DefaultFormatter):
    """XML-style formatting preferred by some models like Claude.

    Uses XML tags for clear structure and semantic meaning.

    Format:
        <system_instructions>
        {content}
        </system_instructions>

        <customer_profile>
        {content}
        </customer_profile>

        <retrieved_documents>
        <document source="policy.pdf" relevance="0.92">
        {content}
        </document>
        </retrieved_documents>

    Args:
        include_token_hints: Whether to include token count in tags. Default: False.

    Examples:
        >>> formatter = XMLFormatter()
        >>> formatted = formatter.format_layer(system_layer)
    """

    # XML tag names for layers
    XML_TAG_MAPPING: dict[str, str] = {
        "system": "system_instructions",
        "user_context": "customer_profile",
        "retrieved_docs": "retrieved_documents",
        "history": "conversation_history",
        "tool_results": "tool_results",
    }

    def __init__(self, include_token_hints: bool = False) -> None:
        super().__init__(
            section_delimiter="",  # Not used in XML format
            chunk_delimiter="",  # Not used in XML format
            include_token_hints=include_token_hints,
        )

    def _get_xml_tag(self, layer_name: str) -> str:
        """Get the XML tag name for a layer.

        Args:
            layer_name: The layer name.

        Returns:
            The XML tag name.
        """
        return self.XML_TAG_MAPPING.get(layer_name, layer_name.replace(" ", "_").lower())

    def format_section_header(self, layer_name: str, token_count: int | None = None) -> str:
        """Format an opening XML tag for a layer.

        Args:
            layer_name: The layer name.
            token_count: Optional token count to include as attribute.

        Returns:
            The opening XML tag.
        """
        tag = self._get_xml_tag(layer_name)
        if self._include_token_hints and token_count is not None:
            return f'<{tag} tokens="{token_count}">'
        return f"<{tag}>"

    def format_layer(self, layer: BaseLayer) -> str:
        """Format a single layer with XML tags.

        Args:
            layer: The layer to format.

        Returns:
            The formatted layer content with XML tags.

        Raises:
            ValueError: If the layer has not been built.
        """
        if layer.built_content is None:
            raise ValueError(f"Layer '{layer.name}' has not been built yet")

        tag = self._get_xml_tag(layer.name)
        token_count = layer.get_token_count() if self._include_token_hints else None

        opening = self.format_section_header(layer.name, token_count)
        closing = f"</{tag}>"

        return f"{opening}\n{layer.built_content}\n{closing}"

    def format_all(self, layers: dict[str, BaseLayer]) -> str:
        """Format all layers into XML structure.

        Args:
            layers: Dict mapping layer names to layer instances.

        Returns:
            The combined formatted XML string.
        """
        formatted_parts = []

        for layer in layers.values():
            if layer.built_content:
                formatted_parts.append(self.format_layer(layer))

        return "\n\n".join(formatted_parts)

    def format_chunk(
        self,
        text: str,
        source: str | None = None,
        score: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Format a retrieved chunk with XML tags.

        Args:
            text: The chunk text.
            source: Optional source identifier.
            score: Optional relevance score.
            metadata: Optional additional metadata.

        Returns:
            The formatted chunk with XML tags.
        """
        attrs = []
        if source:
            attrs.append(f'source="{source}"')
        if score is not None:
            attrs.append(f'relevance="{score:.2f}"')

        attr_str = " " + " ".join(attrs) if attrs else ""
        return f"<document{attr_str}>\n{text}\n</document>"

    def format_message(self, role: str, content: str) -> str:
        """Format a single message with XML tags.

        Args:
            role: The message role.
            content: The message content.

        Returns:
            The formatted message with XML tags.
        """
        return f"<message role=\"{role}\">\n{content}\n</message>"


class MinimalFormatter(DefaultFormatter):
    """Minimal formatting for token-constrained scenarios.

    Uses simple newlines and labels without heavy delimiters.
    Optimized for maximum content with minimum formatting overhead.

    Format:
        [System]
        {content}

        [Context]
        {content}

        [Documents]
        {content}

    Args:
        include_token_hints: Whether to include token counts. Default: False.

    Examples:
        >>> formatter = MinimalFormatter()
        >>> formatted = formatter.format_layer(system_layer)
    """

    # Minimal labels for layers
    MINIMAL_LABELS: dict[str, str] = {
        "system": "System",
        "user_context": "Context",
        "retrieved_docs": "Documents",
        "history": "History",
        "tool_results": "Tools",
    }

    def __init__(self, include_token_hints: bool = False) -> None:
        super().__init__(
            section_delimiter="",
            chunk_delimiter="",
            include_token_hints=include_token_hints,
        )

    def get_section_label(self, layer_name: str) -> str:
        """Get the minimal label for a layer.

        Args:
            layer_name: The layer name.

        Returns:
            The minimal label.
        """
        return self.MINIMAL_LABELS.get(layer_name, layer_name.title())

    def format_section_header(self, layer_name: str, token_count: int | None = None) -> str:
        """Format a minimal section header.

        Args:
            layer_name: The layer name.
            token_count: Optional token count (included if hints enabled).

        Returns:
            The minimal section header.
        """
        label = self.get_section_label(layer_name)
        if self._include_token_hints and token_count is not None:
            return f"[{label} ({token_count})]"
        return f"[{label}]"

    def format_layer(self, layer: BaseLayer) -> str:
        """Format a single layer with minimal overhead.

        Args:
            layer: The layer to format.

        Returns:
            The formatted layer content.

        Raises:
            ValueError: If the layer has not been built.
        """
        if layer.built_content is None:
            raise ValueError(f"Layer '{layer.name}' has not been built yet")

        token_count = layer.get_token_count() if self._include_token_hints else None
        header = self.format_section_header(layer.name, token_count)

        return f"{header}\n{layer.built_content}"

    def format_all(self, layers: dict[str, BaseLayer]) -> str:
        """Format all layers with minimal separators.

        Args:
            layers: Dict mapping layer names to layer instances.

        Returns:
            The combined formatted string.
        """
        formatted_parts = []

        for layer in layers.values():
            if layer.built_content:
                formatted_parts.append(self.format_layer(layer))

        return "\n\n".join(formatted_parts)

    def format_chunk(
        self,
        text: str,
        source: str | None = None,
        score: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Format a retrieved chunk minimally.

        Args:
            text: The chunk text.
            source: Optional source identifier.
            score: Optional relevance score.
            metadata: Optional additional metadata.

        Returns:
            The formatted chunk.
        """
        if source:
            return f"({source})\n{text}"
        return text

    def format_message(self, role: str, content: str) -> str:
        """Format a single message minimally.

        Args:
            role: The message role.
            content: The message content.

        Returns:
            The formatted message.
        """
        # Use single character role prefix
        prefix = role[0].upper() if role else "?"
        return f"{prefix}: {content}"
