"""ContextKit - Composable building blocks for LLM context engineering."""

from llm_contextkit.assembler import ContextAssembler
from llm_contextkit.budget import TokenBudget
from llm_contextkit.exceptions import (
    BudgetExceededError,
    BuildError,
    ContextKitError,
    LayerError,
    TokenizerError,
)
from llm_contextkit.inspector.debug import ContextInspector

__version__ = "0.1.0"

__all__ = [
    "ContextAssembler",
    "ContextInspector",
    "TokenBudget",
    "ContextKitError",
    "BudgetExceededError",
    "LayerError",
    "TokenizerError",
    "BuildError",
]
