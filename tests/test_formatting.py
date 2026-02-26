"""Tests for contextkit.formatting module."""

import pytest

from llm_contextkit.formatting.formatter import DefaultFormatter
from llm_contextkit.formatting.templates import MinimalFormatter, XMLFormatter
from llm_contextkit.layers.system import SystemLayer
from llm_contextkit.tokenizers.counting import ApproximateCounter


@pytest.fixture
def token_counter():
    """Provide an approximate token counter for tests."""
    return ApproximateCounter()


@pytest.fixture
def formatter():
    """Provide a default formatter for tests."""
    return DefaultFormatter()


class TestDefaultFormatterInit:
    """Tests for DefaultFormatter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        formatter = DefaultFormatter()
        assert formatter.section_delimiter == "##"
        assert formatter.chunk_delimiter == "---"

    def test_custom_delimiters(self):
        """Test custom delimiters."""
        formatter = DefaultFormatter(
            section_delimiter="###",
            chunk_delimiter="~~~",
        )
        assert formatter.section_delimiter == "###"
        assert formatter.chunk_delimiter == "~~~"

    def test_include_token_hints(self):
        """Test include_token_hints option."""
        formatter = DefaultFormatter(include_token_hints=True)
        assert formatter._include_token_hints is True


class TestDefaultFormatterSectionLabels:
    """Tests for section label methods."""

    def test_get_default_section_label(self, formatter):
        """Test getting default section labels."""
        assert formatter.get_section_label("system") == "System Instructions"
        assert formatter.get_section_label("user_context") == "Customer Profile"
        assert formatter.get_section_label("history") == "Conversation History"

    def test_get_unknown_section_label(self, formatter):
        """Test getting label for unknown layer."""
        # Should convert snake_case to Title Case
        assert formatter.get_section_label("my_custom_layer") == "My Custom Layer"

    def test_set_section_label(self, formatter):
        """Test setting custom section label."""
        formatter.set_section_label("system", "Custom System Label")
        assert formatter.get_section_label("system") == "Custom System Label"


class TestDefaultFormatterFormatHeader:
    """Tests for format_section_header method."""

    def test_basic_header(self, formatter):
        """Test basic header formatting."""
        header = formatter.format_section_header("system")
        assert header == "## System Instructions"

    def test_header_with_token_count(self):
        """Test header with token count hints."""
        formatter = DefaultFormatter(include_token_hints=True)
        header = formatter.format_section_header("system", token_count=500)
        assert header == "## System Instructions [500 tokens]"


class TestDefaultFormatterFormatLayer:
    """Tests for format_layer method."""

    def test_format_layer(self, formatter, token_counter):
        """Test formatting a layer."""
        layer = SystemLayer(instructions="You are helpful.")
        layer.build(token_counter)

        formatted = formatter.format_layer(layer)
        assert "## System Instructions" in formatted
        assert "You are helpful." in formatted

    def test_format_layer_not_built_raises(self, formatter):
        """Test that formatting unbuilt layer raises error."""
        layer = SystemLayer(instructions="Test")
        # Don't build

        with pytest.raises(ValueError, match="has not been built"):
            formatter.format_layer(layer)


class TestDefaultFormatterFormatAll:
    """Tests for format_all method."""

    def test_format_all_single_layer(self, formatter, token_counter):
        """Test formatting all layers with single layer."""
        layer = SystemLayer(instructions="Be helpful.")
        layer.build(token_counter)

        formatted = formatter.format_all({"system": layer})
        assert "## System Instructions" in formatted
        assert "Be helpful." in formatted

    def test_format_all_multiple_layers(self, formatter, token_counter):
        """Test formatting multiple layers."""
        system = SystemLayer(instructions="Be helpful.", name="system")
        system.build(token_counter)

        custom = SystemLayer(instructions="Custom content", name="custom_layer")
        custom.build(token_counter)

        formatted = formatter.format_all({
            "system": system,
            "custom_layer": custom,
        })

        assert "## System Instructions" in formatted
        assert "Be helpful." in formatted
        assert "## Custom Layer" in formatted
        assert "Custom content" in formatted

    def test_format_all_skips_empty_layers(self, formatter, token_counter):
        """Test that format_all skips layers with no content."""
        layer = SystemLayer(instructions="Test")
        # Don't build - built_content will be None

        formatted = formatter.format_all({"system": layer})
        assert formatted == ""


class TestDefaultFormatterFormatMessage:
    """Tests for format_message method."""

    def test_format_user_message(self, formatter):
        """Test formatting user message."""
        formatted = formatter.format_message("user", "Hello!")
        assert formatted == "User: Hello!"

    def test_format_assistant_message(self, formatter):
        """Test formatting assistant message."""
        formatted = formatter.format_message("assistant", "Hi there!")
        assert formatted == "Assistant: Hi there!"


class TestDefaultFormatterFormatChunk:
    """Tests for format_chunk method."""

    def test_format_chunk_text_only(self, formatter):
        """Test formatting chunk with text only."""
        formatted = formatter.format_chunk("This is the chunk content.")
        assert formatted == "This is the chunk content."

    def test_format_chunk_with_source(self, formatter):
        """Test formatting chunk with source."""
        formatted = formatter.format_chunk(
            "Chunk content",
            source="document.pdf",
        )
        assert "[Source: document.pdf]" in formatted
        assert "Chunk content" in formatted

    def test_format_chunk_with_score(self, formatter):
        """Test formatting chunk with relevance score."""
        formatted = formatter.format_chunk(
            "Chunk content",
            score=0.92,
        )
        assert "[Relevance: 0.92]" in formatted
        assert "Chunk content" in formatted

    def test_format_chunk_with_all_metadata(self, formatter):
        """Test formatting chunk with all metadata."""
        formatted = formatter.format_chunk(
            "Chunk content",
            source="doc.pdf",
            score=0.85,
        )
        assert "[Source: doc.pdf | Relevance: 0.85]" in formatted
        assert "Chunk content" in formatted


# ============================================================================
# XMLFormatter Tests
# ============================================================================


class TestXMLFormatterInit:
    """Tests for XMLFormatter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        formatter = XMLFormatter()
        assert formatter._include_token_hints is False

    def test_with_token_hints(self):
        """Test initialization with token hints."""
        formatter = XMLFormatter(include_token_hints=True)
        assert formatter._include_token_hints is True


class TestXMLFormatterFormatHeader:
    """Tests for XMLFormatter section headers."""

    def test_basic_header(self):
        """Test basic XML opening tag."""
        formatter = XMLFormatter()
        header = formatter.format_section_header("system")
        assert header == "<system_instructions>"

    def test_header_with_token_count(self):
        """Test XML tag with token attribute."""
        formatter = XMLFormatter(include_token_hints=True)
        header = formatter.format_section_header("system", token_count=500)
        assert header == '<system_instructions tokens="500">'

    def test_custom_layer_name(self):
        """Test XML tag for custom layer name."""
        formatter = XMLFormatter()
        header = formatter.format_section_header("my_custom_layer")
        assert header == "<my_custom_layer>"


class TestXMLFormatterFormatLayer:
    """Tests for XMLFormatter.format_layer method."""

    def test_format_layer(self, token_counter):
        """Test formatting a layer with XML tags."""
        formatter = XMLFormatter()
        layer = SystemLayer(instructions="You are helpful.")
        layer.build(token_counter)

        formatted = formatter.format_layer(layer)
        assert "<system_instructions>" in formatted
        assert "</system_instructions>" in formatted
        assert "You are helpful." in formatted

    def test_format_layer_not_built_raises(self):
        """Test that formatting unbuilt layer raises error."""
        formatter = XMLFormatter()
        layer = SystemLayer(instructions="Test")

        with pytest.raises(ValueError, match="has not been built"):
            formatter.format_layer(layer)


class TestXMLFormatterFormatAll:
    """Tests for XMLFormatter.format_all method."""

    def test_format_all(self, token_counter):
        """Test formatting multiple layers."""
        formatter = XMLFormatter()
        layer = SystemLayer(instructions="Be helpful.", name="system")
        layer.build(token_counter)

        formatted = formatter.format_all({"system": layer})
        assert "<system_instructions>" in formatted
        assert "</system_instructions>" in formatted


class TestXMLFormatterFormatChunk:
    """Tests for XMLFormatter.format_chunk method."""

    def test_format_chunk_basic(self):
        """Test basic chunk formatting."""
        formatter = XMLFormatter()
        formatted = formatter.format_chunk("Chunk content")
        assert "<document>" in formatted
        assert "</document>" in formatted
        assert "Chunk content" in formatted

    def test_format_chunk_with_metadata(self):
        """Test chunk with metadata attributes."""
        formatter = XMLFormatter()
        formatted = formatter.format_chunk(
            "Chunk content",
            source="doc.pdf",
            score=0.85,
        )
        assert 'source="doc.pdf"' in formatted
        assert 'relevance="0.85"' in formatted


class TestXMLFormatterFormatMessage:
    """Tests for XMLFormatter.format_message method."""

    def test_format_message(self):
        """Test message formatting."""
        formatter = XMLFormatter()
        formatted = formatter.format_message("user", "Hello!")
        assert '<message role="user">' in formatted
        assert "</message>" in formatted
        assert "Hello!" in formatted


# ============================================================================
# MinimalFormatter Tests
# ============================================================================


class TestMinimalFormatterInit:
    """Tests for MinimalFormatter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        formatter = MinimalFormatter()
        assert formatter._include_token_hints is False

    def test_with_token_hints(self):
        """Test initialization with token hints."""
        formatter = MinimalFormatter(include_token_hints=True)
        assert formatter._include_token_hints is True


class TestMinimalFormatterSectionLabels:
    """Tests for MinimalFormatter section labels."""

    def test_get_section_label(self):
        """Test minimal section labels."""
        formatter = MinimalFormatter()
        assert formatter.get_section_label("system") == "System"
        assert formatter.get_section_label("user_context") == "Context"
        assert formatter.get_section_label("history") == "History"

    def test_unknown_layer_label(self):
        """Test label for unknown layer."""
        formatter = MinimalFormatter()
        assert formatter.get_section_label("custom_layer") == "Custom_Layer"


class TestMinimalFormatterFormatHeader:
    """Tests for MinimalFormatter section headers."""

    def test_basic_header(self):
        """Test basic minimal header."""
        formatter = MinimalFormatter()
        header = formatter.format_section_header("system")
        assert header == "[System]"

    def test_header_with_token_count(self):
        """Test header with token count."""
        formatter = MinimalFormatter(include_token_hints=True)
        header = formatter.format_section_header("system", token_count=500)
        assert header == "[System (500)]"


class TestMinimalFormatterFormatLayer:
    """Tests for MinimalFormatter.format_layer method."""

    def test_format_layer(self, token_counter):
        """Test formatting a layer minimally."""
        formatter = MinimalFormatter()
        layer = SystemLayer(instructions="You are helpful.")
        layer.build(token_counter)

        formatted = formatter.format_layer(layer)
        assert "[System]" in formatted
        assert "You are helpful." in formatted
        # Should NOT have ## or XML tags
        assert "##" not in formatted
        assert "<" not in formatted

    def test_format_layer_not_built_raises(self):
        """Test that formatting unbuilt layer raises error."""
        formatter = MinimalFormatter()
        layer = SystemLayer(instructions="Test")

        with pytest.raises(ValueError, match="has not been built"):
            formatter.format_layer(layer)


class TestMinimalFormatterFormatAll:
    """Tests for MinimalFormatter.format_all method."""

    def test_format_all(self, token_counter):
        """Test formatting multiple layers minimally."""
        formatter = MinimalFormatter()
        layer = SystemLayer(instructions="Be helpful.", name="system")
        layer.build(token_counter)

        formatted = formatter.format_all({"system": layer})
        assert "[System]" in formatted
        assert "Be helpful." in formatted


class TestMinimalFormatterFormatChunk:
    """Tests for MinimalFormatter.format_chunk method."""

    def test_format_chunk_basic(self):
        """Test basic chunk formatting."""
        formatter = MinimalFormatter()
        formatted = formatter.format_chunk("Chunk content")
        assert formatted == "Chunk content"

    def test_format_chunk_with_source(self):
        """Test chunk with source."""
        formatter = MinimalFormatter()
        formatted = formatter.format_chunk("Chunk content", source="doc.pdf")
        assert "(doc.pdf)" in formatted
        assert "Chunk content" in formatted


class TestMinimalFormatterFormatMessage:
    """Tests for MinimalFormatter.format_message method."""

    def test_format_user_message(self):
        """Test user message formatting."""
        formatter = MinimalFormatter()
        formatted = formatter.format_message("user", "Hello!")
        assert formatted == "U: Hello!"

    def test_format_assistant_message(self):
        """Test assistant message formatting."""
        formatter = MinimalFormatter()
        formatted = formatter.format_message("assistant", "Hi!")
        assert formatted == "A: Hi!"
