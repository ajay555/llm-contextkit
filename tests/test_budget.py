"""Tests for contextkit.budget module."""

import pytest

from llm_contextkit.budget import TokenBudget
from llm_contextkit.exceptions import BudgetExceededError


class TestTokenBudgetInit:
    """Tests for TokenBudget initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        budget = TokenBudget(total=4096)
        assert budget.total == 4096
        assert budget.reserve_for_output == 1000  # default

    def test_custom_reserve(self):
        """Test custom reserve_for_output."""
        budget = TokenBudget(total=4096, reserve_for_output=500)
        assert budget.reserve_for_output == 500

    def test_negative_total_raises(self):
        """Test that negative total raises ValueError."""
        with pytest.raises(ValueError, match="total must be non-negative"):
            TokenBudget(total=-1)

    def test_negative_reserve_raises(self):
        """Test that negative reserve raises ValueError."""
        with pytest.raises(ValueError, match="reserve_for_output must be non-negative"):
            TokenBudget(total=4096, reserve_for_output=-1)

    def test_reserve_exceeds_total_raises(self):
        """Test that reserve > total raises ValueError."""
        with pytest.raises(ValueError, match="reserve_for_output cannot exceed total"):
            TokenBudget(total=1000, reserve_for_output=2000)


class TestTokenBudgetAllocate:
    """Tests for TokenBudget.allocate method."""

    def test_basic_allocation(self):
        """Test basic allocation."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        assert budget.get_allocation("system") == 500

    def test_allocation_priority(self):
        """Test allocation with priority."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        budget.allocate("history", 1000, priority=5)
        assert budget.get_priority("system") == 10
        assert budget.get_priority("history") == 5

    def test_multiple_allocations(self):
        """Test multiple allocations."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500)
        budget.allocate("history", 1500)
        budget.allocate("context", 500)
        assert budget.get_total_allocated() == 2500

    def test_negative_tokens_raises(self):
        """Test that negative tokens raises ValueError."""
        budget = TokenBudget(total=4096)
        with pytest.raises(ValueError, match="tokens must be non-negative"):
            budget.allocate("system", -100)

    def test_over_allocation_raises(self):
        """Test that over-allocation raises BudgetExceededError."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        # Available: 4096 - 1000 = 3096
        budget.allocate("system", 3000)
        with pytest.raises(BudgetExceededError):
            budget.allocate("history", 500)  # Would exceed

    def test_update_existing_allocation(self):
        """Test updating an existing allocation."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500)
        budget.allocate("system", 600)  # Update
        assert budget.get_allocation("system") == 600


class TestTokenBudgetAvailable:
    """Tests for TokenBudget.get_available method."""

    def test_initial_available(self):
        """Test available tokens before any allocation."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        # 4096 - 1000 = 3096 available
        assert budget.get_available() == 3096

    def test_available_after_allocation(self):
        """Test available tokens after allocation."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500)
        assert budget.get_available() == 2596


class TestTokenBudgetUsage:
    """Tests for TokenBudget usage tracking."""

    def test_set_and_get_used(self):
        """Test setting and getting used tokens."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500)
        budget.set_used("system", 350)
        assert budget.get_used("system") == 350

    def test_get_used_nonexistent(self):
        """Test getting used for non-existent layer."""
        budget = TokenBudget(total=4096)
        assert budget.get_used("nonexistent") == 0

    def test_get_total_used(self):
        """Test getting total used across layers."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500)
        budget.allocate("history", 1000)
        budget.set_used("system", 400)
        budget.set_used("history", 800)
        assert budget.get_total_used() == 1200


class TestTokenBudgetCounting:
    """Tests for TokenBudget.count_tokens method."""

    def test_count_tokens(self):
        """Test token counting."""
        budget = TokenBudget(total=4096, tokenizer="approximate")
        count = budget.count_tokens("hello world")
        assert count > 0

    def test_count_empty_string(self):
        """Test counting empty string."""
        budget = TokenBudget(total=4096, tokenizer="approximate")
        assert budget.count_tokens("") == 0


class TestTokenBudgetPriority:
    """Tests for TokenBudget priority-related methods."""

    def test_get_layers_by_priority(self):
        """Test getting layers sorted by priority."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("history", 500, priority=5)
        budget.allocate("system", 500, priority=10)
        budget.allocate("context", 500, priority=3)

        layers = budget.get_layers_by_priority()
        assert layers == ["context", "history", "system"]

    def test_get_priority_nonexistent(self):
        """Test getting priority for non-existent layer."""
        budget = TokenBudget(total=4096)
        assert budget.get_priority("nonexistent") == 0


class TestTokenBudgetSummary:
    """Tests for TokenBudget.summary method."""

    def test_summary(self):
        """Test summary generation."""
        budget = TokenBudget(total=4096, reserve_for_output=1000)
        budget.allocate("system", 500, priority=10)
        budget.allocate("history", 1000, priority=5)
        budget.set_used("system", 400)
        budget.set_used("history", 800)

        summary = budget.summary()

        assert summary["total"] == 4096
        assert summary["reserved"] == 1000
        assert summary["available_for_context"] == 3096
        assert summary["total_allocated"] == 1500
        assert summary["total_used"] == 1200
        assert summary["unallocated"] == 1596

        assert "system" in summary["layers"]
        assert summary["layers"]["system"]["allocated"] == 500
        assert summary["layers"]["system"]["used"] == 400
        assert summary["layers"]["system"]["priority"] == 10
