"""Tests for contextkit.exceptions module."""

import pytest

from llm_contextkit.exceptions import (
    BudgetExceededError,
    BuildError,
    ContextKitError,
    LayerError,
    TokenizerError,
)


class TestContextKitError:
    """Tests for base ContextKitError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ContextKitError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_inheritance(self):
        """Test that ContextKitError inherits from Exception."""
        error = ContextKitError("test")
        assert isinstance(error, Exception)


class TestBudgetExceededError:
    """Tests for BudgetExceededError."""

    def test_basic_error(self):
        """Test basic budget exceeded error."""
        error = BudgetExceededError("Budget exceeded")
        assert str(error) == "Budget exceeded"
        assert error.layer_name is None
        assert error.requested_tokens is None
        assert error.available_tokens is None

    def test_with_details(self):
        """Test error with full details."""
        error = BudgetExceededError(
            "System layer too large",
            layer_name="system",
            requested_tokens=1000,
            available_tokens=500,
        )
        assert error.layer_name == "system"
        assert error.requested_tokens == 1000
        assert error.available_tokens == 500

    def test_inheritance(self):
        """Test that BudgetExceededError inherits from ContextKitError."""
        error = BudgetExceededError("test")
        assert isinstance(error, ContextKitError)


class TestLayerError:
    """Tests for LayerError."""

    def test_basic_error(self):
        """Test basic layer error."""
        error = LayerError("Invalid layer config")
        assert str(error) == "Invalid layer config"
        assert error.layer_name is None

    def test_with_layer_name(self):
        """Test error with layer name."""
        error = LayerError("Missing required field", layer_name="history")
        assert error.layer_name == "history"

    def test_inheritance(self):
        """Test inheritance."""
        error = LayerError("test")
        assert isinstance(error, ContextKitError)


class TestTokenizerError:
    """Tests for TokenizerError."""

    def test_basic_error(self):
        """Test basic tokenizer error."""
        error = TokenizerError("Tokenizer failed to load")
        assert str(error) == "Tokenizer failed to load"
        assert error.tokenizer_name is None

    def test_with_tokenizer_name(self):
        """Test error with tokenizer name."""
        error = TokenizerError("Unknown encoding", tokenizer_name="cl100k")
        assert error.tokenizer_name == "cl100k"

    def test_inheritance(self):
        """Test inheritance."""
        error = TokenizerError("test")
        assert isinstance(error, ContextKitError)


class TestBuildError:
    """Tests for BuildError."""

    def test_basic_error(self):
        """Test basic build error."""
        error = BuildError("Build failed")
        assert str(error) == "Build failed"
        assert error.phase is None

    def test_with_phase(self):
        """Test error with phase."""
        error = BuildError("Truncation failed", phase="truncation")
        assert error.phase == "truncation"

    def test_inheritance(self):
        """Test inheritance."""
        error = BuildError("test")
        assert isinstance(error, ContextKitError)


class TestExceptionCatching:
    """Test that all exceptions can be caught with ContextKitError."""

    def test_catch_all_with_base(self):
        """Test catching all exceptions with base class."""
        exceptions = [
            BudgetExceededError("test"),
            LayerError("test"),
            TokenizerError("test"),
            BuildError("test"),
        ]

        for exc in exceptions:
            with pytest.raises(ContextKitError):
                raise exc
