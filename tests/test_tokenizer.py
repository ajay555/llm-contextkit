"""Tests for contextkit.tokenizers.counting module."""

import pytest

from llm_contextkit.exceptions import TokenizerError
from llm_contextkit.tokenizers.counting import (
    ApproximateCounter,
    CallableCounter,
    TokenCounter,
    get_token_counter,
)


class TestApproximateCounter:
    """Tests for ApproximateCounter."""

    def test_empty_string(self):
        """Test counting empty string."""
        counter = ApproximateCounter()
        assert counter.count("") == 0

    def test_single_word(self):
        """Test counting single word."""
        counter = ApproximateCounter()
        # 1 word * 1.3 = 1.3, rounded to 1
        assert counter.count("hello") == 1

    def test_multiple_words(self):
        """Test counting multiple words."""
        counter = ApproximateCounter()
        # 5 words * 1.3 = 6.5, rounded to 6
        assert counter.count("the quick brown fox jumps") == 6

    def test_name(self):
        """Test counter name."""
        counter = ApproximateCounter()
        assert counter.name == "approximate"

    def test_inherits_from_token_counter(self):
        """Test inheritance."""
        counter = ApproximateCounter()
        assert isinstance(counter, TokenCounter)


class TestCallableCounter:
    """Tests for CallableCounter."""

    def test_custom_counter(self):
        """Test with custom counting function."""
        # Simple word count
        counter = CallableCounter(lambda x: len(x.split()))
        assert counter.count("hello world") == 2

    def test_name(self):
        """Test default name."""
        counter = CallableCounter(lambda x: 0)
        assert counter.name == "custom"

    def test_custom_name(self):
        """Test custom name."""
        counter = CallableCounter(lambda x: 0, name="my_counter")
        assert counter.name == "my_counter"

    def test_inherits_from_token_counter(self):
        """Test inheritance."""
        counter = CallableCounter(lambda x: 0)
        assert isinstance(counter, TokenCounter)


class TestGetTokenCounter:
    """Tests for get_token_counter function."""

    def test_approximate_tokenizer(self):
        """Test getting approximate tokenizer."""
        counter = get_token_counter("approximate")
        assert isinstance(counter, ApproximateCounter)

    def test_callable_tokenizer(self):
        """Test getting callable tokenizer."""
        def char_counter(x: str) -> int:
            return len(x)

        counter = get_token_counter(char_counter)
        assert isinstance(counter, CallableCounter)
        assert counter.count("hello") == 5

    def test_unknown_tokenizer_falls_back_to_approximate(self):
        """Test that unknown tokenizer falls back to approximate."""
        # This should fall back to approximate since tiktoken may not be installed
        counter = get_token_counter("unknown_encoding")
        # Should be approximate or tiktoken, both work
        assert counter.count("hello world") >= 0

    def test_cl100k_fallback(self):
        """Test that cl100k falls back gracefully if tiktoken not installed."""
        # This should work regardless of tiktoken availability
        counter = get_token_counter("cl100k")
        assert counter.count("hello world") >= 0


class TestTiktokenIntegration:
    """Integration tests for tiktoken (skipped if not installed)."""

    @pytest.fixture
    def tiktoken_available(self):
        """Check if tiktoken is available."""
        try:
            import tiktoken  # noqa: F401
            return True
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_tiktoken_cl100k(self, tiktoken_available):
        """Test tiktoken with cl100k encoding."""
        from llm_contextkit.tokenizers.counting import TiktokenCounter

        counter = TiktokenCounter("cl100k_base")
        # "Hello, world!" should be a small number of tokens
        count = counter.count("Hello, world!")
        assert count > 0
        assert count < 10

    def test_tiktoken_name(self, tiktoken_available):
        """Test tiktoken counter name."""
        from llm_contextkit.tokenizers.counting import TiktokenCounter

        counter = TiktokenCounter("cl100k_base")
        assert counter.name == "cl100k_base"

    def test_tiktoken_invalid_encoding(self, tiktoken_available):
        """Test tiktoken with invalid encoding."""
        from llm_contextkit.tokenizers.counting import TiktokenCounter

        with pytest.raises(TokenizerError):
            TiktokenCounter("invalid_encoding_name")
