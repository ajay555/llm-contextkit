# Contributing to LLM ContextKit

Thank you for your interest in contributing to ContextKit! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and inclusive. We're all here to build something useful together.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-contextkit.git
   cd llm-contextkit
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify setup**
   ```bash
   pytest tests/ -v
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=llm_contextkit --cov-report=term-missing

# Run specific test file
pytest tests/test_assembler.py -v

# Run specific test
pytest tests/test_assembler.py::TestContextAssemblerBuild::test_basic_build -v
```

### Code Quality

We use `ruff` for linting and `mypy` for type checking:

```bash
# Lint code
ruff check llm_contextkit/ tests/

# Auto-fix lint issues
ruff check --fix llm_contextkit/ tests/

# Type check
mypy llm_contextkit/
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/ajay555/llm-contextkit/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Minimal code example if possible

### Suggesting Features

1. Check existing issues and discussions for similar suggestions
2. Create a new issue with:
   - Clear description of the feature
   - Use case and motivation
   - Proposed API (if applicable)

### Submitting Pull Requests

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests and linting**
   ```bash
   pytest tests/ -v
   ruff check llm_contextkit/ tests/
   mypy llm_contextkit/
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: brief description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Style

- **Type hints**: All public functions and methods must have type hints
- **Docstrings**: Use Google-style docstrings for all public classes and methods
- **Line length**: 100 characters max
- **Imports**: Use absolute imports, sorted by `ruff`

### Example

```python
def count_tokens(self, text: str) -> int:
    """Count tokens in the given text.

    Args:
        text: The text to count tokens in.

    Returns:
        The number of tokens in the text.

    Raises:
        TokenizerError: If tokenization fails.
    """
    return self._tokenizer.encode(text)
```

### Testing Style

- One test class per major component
- Clear test method names: `test_<what>_<condition>_<expected>`
- Use fixtures for shared setup
- Test both success and error cases

```python
class TestTokenBudget:
    def test_allocate_valid_tokens_succeeds(self):
        """Test that valid allocation works."""
        budget = TokenBudget(total=4096)
        budget.allocate("system", 500)
        assert budget.get_allocation("system") == 500

    def test_allocate_negative_tokens_raises(self):
        """Test that negative allocation raises ValueError."""
        budget = TokenBudget(total=4096)
        with pytest.raises(ValueError):
            budget.allocate("system", -100)
```

## Architecture Guidelines

### Adding a New Layer Type

1. Create a new file in `llm_contextkit/layers/`
2. Inherit from `BaseLayer`
3. Implement `build()` and `truncate()` methods
4. Add to `llm_contextkit/layers/__init__.py`
5. Write comprehensive tests
6. Add example usage to documentation

### Adding a New History Strategy

1. Add to `llm_contextkit/history/strategies.py`
2. Inherit from `HistoryStrategy`
3. Implement `apply()` method
4. Register in `get_strategy()` function
5. Write tests covering edge cases

## Documentation

- Update README.md for user-facing changes
- Add docstrings for all public API
- Include code examples where helpful
- Keep the API Reference section updated

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release commit and tag
4. Build and publish to PyPI

## Questions?

- Open a GitHub Discussion for general questions
- Open an Issue for bugs or feature requests
- Tag maintainers in PRs if you need review

Thank you for contributing!
