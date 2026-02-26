"""Custom exceptions for ContextKit.

All exceptions inherit from ContextKitError for easy catching of any
library-specific error.
"""

from __future__ import annotations


class ContextKitError(Exception):
    """Base exception for all ContextKit errors.

    Args:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class BudgetExceededError(ContextKitError):
    """Raised when context cannot fit within the token budget even after truncation.

    This typically occurs when high-priority layers (like system prompts) exceed
    their allocation and cannot be truncated further.

    Args:
        message: Human-readable error description.
        layer_name: Name of the layer that exceeded the budget.
        requested_tokens: Number of tokens requested.
        available_tokens: Number of tokens available.
    """

    def __init__(
        self,
        message: str,
        layer_name: str | None = None,
        requested_tokens: int | None = None,
        available_tokens: int | None = None,
    ) -> None:
        self.layer_name = layer_name
        self.requested_tokens = requested_tokens
        self.available_tokens = available_tokens
        super().__init__(message)


class LayerError(ContextKitError):
    """Raised when there is an invalid layer configuration.

    Args:
        message: Human-readable error description.
        layer_name: Name of the layer with the configuration error.
    """

    def __init__(self, message: str, layer_name: str | None = None) -> None:
        self.layer_name = layer_name
        super().__init__(message)


class TokenizerError(ContextKitError):
    """Raised when tokenizer initialization or operation fails.

    Args:
        message: Human-readable error description.
        tokenizer_name: Name of the tokenizer that failed.
    """

    def __init__(self, message: str, tokenizer_name: str | None = None) -> None:
        self.tokenizer_name = tokenizer_name
        super().__init__(message)


class BuildError(ContextKitError):
    """Raised when context assembly fails.

    Args:
        message: Human-readable error description.
        phase: The build phase where the error occurred (e.g., "layer_build", "truncation").
    """

    def __init__(self, message: str, phase: str | None = None) -> None:
        self.phase = phase
        super().__init__(message)
