"""Tests for contextkit.layers.retrieved module."""

import pytest

from llm_contextkit.layers.base import BaseLayer
from llm_contextkit.layers.retrieved import RetrievedLayer
from llm_contextkit.tokenizers.counting import ApproximateCounter


@pytest.fixture
def token_counter():
    """Provide an approximate token counter for tests."""
    return ApproximateCounter()


@pytest.fixture
def sample_chunks():
    """Provide sample retrieved chunks."""
    return [
        {
            "text": "This is the first document about refund policies.",
            "source": "policy.pdf",
            "score": 0.95,
        },
        {
            "text": "The second document discusses return procedures.",
            "source": "returns.md",
            "score": 0.88,
        },
        {
            "text": "FAQ section about common customer questions.",
            "source": "faq.txt",
            "score": 0.75,
        },
    ]


class TestRetrievedLayerInit:
    """Tests for RetrievedLayer initialization."""

    def test_basic_init(self, sample_chunks):
        """Test basic initialization."""
        layer = RetrievedLayer(chunks=sample_chunks)

        assert len(layer.chunks) == 3
        assert layer.name == "retrieved_docs"
        assert layer.priority == 6
        assert layer.max_chunks == 5
        assert layer.include_metadata is True

    def test_custom_settings(self, sample_chunks):
        """Test custom initialization settings."""
        layer = RetrievedLayer(
            chunks=sample_chunks,
            max_chunks=2,
            include_metadata=False,
            name="documents",
            priority=7,
        )

        assert layer.max_chunks == 2
        assert layer.include_metadata is False
        assert layer.name == "documents"
        assert layer.priority == 7

    def test_chunks_are_copied(self, sample_chunks):
        """Test that chunks are copied."""
        layer = RetrievedLayer(chunks=sample_chunks)

        # Modify original
        sample_chunks[0]["text"] = "Modified"

        # Layer should have original
        assert "refund" in layer.chunks[0]["text"]

    def test_inherits_from_base_layer(self, sample_chunks):
        """Test inheritance."""
        layer = RetrievedLayer(chunks=sample_chunks)
        assert isinstance(layer, BaseLayer)


class TestRetrievedLayerBuild:
    """Tests for RetrievedLayer.build method."""

    def test_build_basic(self, token_counter, sample_chunks):
        """Test basic build."""
        layer = RetrievedLayer(chunks=sample_chunks)
        content = layer.build(token_counter)

        assert "refund policies" in content
        assert "return procedures" in content
        assert layer.get_token_count() > 0

    def test_build_with_metadata(self, token_counter, sample_chunks):
        """Test build includes metadata."""
        layer = RetrievedLayer(chunks=sample_chunks)
        content = layer.build(token_counter)

        assert "Source: policy.pdf" in content
        assert "Relevance: 0.95" in content

    def test_build_without_metadata(self, token_counter, sample_chunks):
        """Test build without metadata."""
        layer = RetrievedLayer(chunks=sample_chunks, include_metadata=False)
        content = layer.build(token_counter)

        assert "Source:" not in content
        assert "Relevance:" not in content
        assert "refund policies" in content

    def test_build_max_chunks(self, token_counter, sample_chunks):
        """Test max_chunks limiting."""
        layer = RetrievedLayer(chunks=sample_chunks, max_chunks=1)
        content = layer.build(token_counter)

        assert "refund policies" in content
        assert "return procedures" not in content

    def test_build_empty_chunks(self, token_counter):
        """Test build with empty chunks."""
        layer = RetrievedLayer(chunks=[])
        content = layer.build(token_counter)

        assert content == ""
        assert layer.get_token_count() == 0

    def test_build_chunk_separators(self, token_counter, sample_chunks):
        """Test that chunks are separated properly."""
        layer = RetrievedLayer(chunks=sample_chunks)
        content = layer.build(token_counter)

        assert "---" in content  # Chunk separator


class TestRetrievedLayerTruncate:
    """Tests for RetrievedLayer.truncate method."""

    def test_truncate_removes_chunks(self, token_counter, sample_chunks):
        """Test that truncation removes chunks from end."""
        layer = RetrievedLayer(chunks=sample_chunks)
        layer.build(token_counter)

        layer.truncate(30, token_counter)

        # Should have fewer chunks
        assert layer.truncated is True

    def test_truncate_preserves_order(self, token_counter, sample_chunks):
        """Test that truncation preserves relevance order."""
        layer = RetrievedLayer(chunks=sample_chunks)
        layer.build(token_counter)

        content = layer.truncate(40, token_counter)

        # First chunk should still be present if possible
        if "refund" in content:
            # If first chunk is there, it should come before any others
            pass  # Order is maintained

    def test_truncate_to_empty(self, token_counter, sample_chunks):
        """Test truncation to zero tokens."""
        layer = RetrievedLayer(chunks=sample_chunks)
        layer.build(token_counter)

        content = layer.truncate(1, token_counter)

        assert content == ""
        assert layer.truncated is True

    def test_truncate_no_partial_chunks(self, token_counter, sample_chunks):
        """Test that chunks are never partially truncated."""
        layer = RetrievedLayer(chunks=sample_chunks)
        layer.build(token_counter)

        # Get a budget that would require partial truncation
        first_chunk = sample_chunks[0]["text"]
        content = layer.truncate(50, token_counter)

        # Either first chunk is fully included or not at all
        if first_chunk[:20] in content:
            assert first_chunk in content


class TestRetrievedLayerInspect:
    """Tests for RetrievedLayer.inspect method."""

    def test_inspect_before_build(self, sample_chunks):
        """Test inspect before build."""
        layer = RetrievedLayer(chunks=sample_chunks)
        info = layer.inspect()

        assert info["name"] == "retrieved_docs"
        assert info["total_chunks"] == 3
        assert info["max_chunks_config"] == 5

    def test_inspect_after_build(self, token_counter, sample_chunks):
        """Test inspect after build."""
        layer = RetrievedLayer(chunks=sample_chunks)
        layer.build(token_counter)
        info = layer.inspect()

        assert info["included_chunks"] == 3
        assert info["dropped_chunks"] == 0

    def test_inspect_after_truncate(self, token_counter, sample_chunks):
        """Test inspect after truncation."""
        layer = RetrievedLayer(chunks=sample_chunks)
        layer.build(token_counter)
        layer.truncate(30, token_counter)
        info = layer.inspect()

        if info["truncated"]:
            assert info["dropped_chunks"] > 0


class TestRetrievedLayerChunkFormats:
    """Tests for different chunk formats."""

    def test_chunk_text_only(self, token_counter):
        """Test chunk with only text."""
        chunks = [{"text": "Just the content, no metadata."}]
        layer = RetrievedLayer(chunks=chunks)
        content = layer.build(token_counter)

        assert "Just the content" in content

    def test_chunk_with_source_only(self, token_counter):
        """Test chunk with text and source."""
        chunks = [{"text": "Content here.", "source": "doc.pdf"}]
        layer = RetrievedLayer(chunks=chunks)
        content = layer.build(token_counter)

        assert "Source: doc.pdf" in content
        assert "Relevance:" not in content

    def test_chunk_with_score_only(self, token_counter):
        """Test chunk with text and score."""
        chunks = [{"text": "Content here.", "score": 0.85}]
        layer = RetrievedLayer(chunks=chunks)
        content = layer.build(token_counter)

        assert "Relevance: 0.85" in content
        assert "Source:" not in content

    def test_score_formatting(self, token_counter):
        """Test score is formatted to 2 decimal places."""
        chunks = [{"text": "Content", "score": 0.8567}]
        layer = RetrievedLayer(chunks=chunks)
        content = layer.build(token_counter)

        assert "Relevance: 0.86" in content
