# LLM ContextKit

**Composable building blocks for LLM context engineering.**

[![PyPI version](https://img.shields.io/pypi/v/llm-contextkit.svg)](https://pypi.org/project/llm-contextkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-contextkit.svg)](https://pypi.org/project/llm-contextkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ajay555/llm-contextkit/actions/workflows/tests.yml/badge.svg)](https://github.com/ajay555/llm-contextkit/actions)

---

## The Problem

Every team building production GenAI applications ends up building the same context management infrastructure from scratch:

- How do you assemble multiple layers of context (system prompts, user context, conversation history, retrieved documents, tool results) into a single coherent payload?
- How do you manage token budgets so the context fits within model limits?
- How do you handle conversation history (sliding window, summarization, selective inclusion)?
- How do you debug what actually went into the context window when something goes wrong?

There's no focused, well-designed library that owns this layer cleanly. LangChain gives you orchestration, vector databases give you retrieval, but nobody owns the **"context assembly and management"** layer.

**LLM ContextKit fills that gap.**

---

## Quick Start

```bash
pip install llm-contextkit
```

```python
from llm_contextkit import ContextAssembler, TokenBudget
from llm_contextkit.layers import SystemLayer, HistoryLayer

# Create a token budget
budget = TokenBudget(total=4096, reserve_for_output=1000)
budget.allocate("system", 500, priority=10)
budget.allocate("history", 2500, priority=5)

# Assemble context
assembler = ContextAssembler(budget=budget)
assembler.add_layer(
    SystemLayer(instructions="You are a helpful assistant.")
)
assembler.add_layer(
    HistoryLayer(messages=conversation_history, strategy="sliding_window")
)

# Build for your LLM provider
messages = assembler.build_for_openai()
# Or: payload = assembler.build_for_anthropic()

# Debug what was built
print(assembler.inspect_pretty())
```

---

## Key Features

- **Token Budget Management** — Allocate tokens across layers, automatic truncation by priority
- **Composable Context Layers** — System prompts, user context, history, RAG chunks, tool results
- **Multiple History Strategies** — Sliding window, summarization, selective inclusion
- **Context Inspection & Debugging** — See exactly what went into your context and why
- **OpenAI + Anthropic Output Formats** — Build once, deploy to any provider
- **Zero Required Dependencies** — Works with just Python stdlib; tiktoken optional for accurate counting

---

## Examples

### Basic Chatbot

```python
from llm_contextkit import ContextAssembler, TokenBudget
from llm_contextkit.layers import SystemLayer, HistoryLayer

budget = TokenBudget(total=4096, reserve_for_output=1000)
budget.allocate("system", 500, priority=10)
budget.allocate("history", 2500, priority=5)

assembler = ContextAssembler(budget=budget)
assembler.add_layer(
    SystemLayer(
        instructions="You are a helpful customer support agent.",
        few_shot_examples=[
            {"input": "I can't log in", "output": "I'd be happy to help with your login issue..."}
        ]
    )
)
assembler.add_layer(
    HistoryLayer(
        messages=conversation_history,
        strategy="sliding_window",
        strategy_config={"max_turns": 10}
    )
)

messages = assembler.build_for_openai()
```

### RAG Application

```python
from llm_contextkit import ContextAssembler, TokenBudget
from llm_contextkit.layers import SystemLayer, UserContextLayer, HistoryLayer, RetrievedLayer

budget = TokenBudget(total=8000, reserve_for_output=1500)
budget.allocate("system", 500, priority=10)
budget.allocate("user_context", 300, priority=7)
budget.allocate("retrieved_docs", 3500, priority=6)
budget.allocate("history", 2000, priority=5)

assembler = ContextAssembler(budget=budget)
assembler.add_layer(SystemLayer(instructions="Answer based only on the provided documents."))
assembler.add_layer(UserContextLayer(context={"name": "Jane", "tier": "Enterprise"}))
assembler.add_layer(RetrievedLayer(chunks=retrieved_chunks, max_chunks=5))
assembler.add_layer(HistoryLayer(messages=history, strategy="sliding_window_with_summary"))

payload = assembler.build_for_anthropic()
```

### Agentic Workflow

```python
from llm_contextkit import ContextAssembler, TokenBudget
from llm_contextkit.layers import SystemLayer, HistoryLayer, RetrievedLayer, ToolResultsLayer

budget = TokenBudget(total=16000, reserve_for_output=2000)
budget.allocate("system", 800, priority=10)
budget.allocate("tool_results", 4000, priority=8)
budget.allocate("retrieved_docs", 4000, priority=6)
budget.allocate("history", 5000, priority=5)

assembler = ContextAssembler(budget=budget)
assembler.add_layer(SystemLayer(instructions="You are a research agent with tool access."))
assembler.add_layer(ToolResultsLayer(results=tool_outputs, include_inputs=True))
assembler.add_layer(RetrievedLayer(chunks=context_docs))
assembler.add_layer(HistoryLayer(messages=agent_history))

messages = assembler.build_for_openai()
print(assembler.inspect_pretty())
```

### Context Inspection

```python
from llm_contextkit import ContextInspector

inspector = ContextInspector(tokenizer="cl100k")

# Analyze any messages payload
report = inspector.analyze(messages)
print(report.pretty())

# Compare two payloads
diff = inspector.diff(before_messages, after_messages)
print(diff.pretty())
```

---

## Context Layers

| Layer | Purpose | Default Priority |
|-------|---------|------------------|
| `SystemLayer` | System instructions + few-shot examples | 10 (highest) |
| `ToolResultsLayer` | Tool/API call results for agents | 8 |
| `UserContextLayer` | User metadata and session context | 7 |
| `RetrievedLayer` | RAG chunks with source/relevance metadata | 6 |
| `HistoryLayer` | Conversation history with strategies | 5 (lowest) |

Lower priority layers are truncated first when the context exceeds the budget.

---

## History Strategies

```python
# Keep last N turns
HistoryLayer(messages, strategy="sliding_window", strategy_config={"max_turns": 10})

# Summarize older turns, keep recent in full
HistoryLayer(messages, strategy="sliding_window_with_summary",
             strategy_config={"max_recent_turns": 5, "summarizer": my_summarizer})

# Include only messages relevant to current query
HistoryLayer(messages, strategy="selective",
             strategy_config={"query": user_query, "relevance_threshold": 0.5})
```

---

## Formatters

```python
from llm_contextkit.formatting import DefaultFormatter, XMLFormatter, MinimalFormatter

# Markdown-style (default)
assembler = ContextAssembler(budget, formatter=DefaultFormatter())

# XML tags (preferred by Claude)
assembler = ContextAssembler(budget, formatter=XMLFormatter())

# Minimal overhead for token-constrained scenarios
assembler = ContextAssembler(budget, formatter=MinimalFormatter())
```

---

## API Reference

### TokenBudget

```python
TokenBudget(
    total=4096,              # Total context window
    tokenizer="cl100k",      # "cl100k", "o200k", "approximate", or callable
    reserve_for_output=1000  # Reserve for model response
)
budget.allocate(layer_name, tokens, priority=0)
budget.count_tokens(text)
budget.summary()
```

### ContextAssembler

```python
ContextAssembler(budget, formatter=None)
assembler.add_layer(layer)
assembler.remove_layer(name)
assembler.build()                # Returns generic dict
assembler.build_for_openai()     # Returns OpenAI messages format
assembler.build_for_anthropic()  # Returns Anthropic API format
assembler.inspect()              # Returns build metadata
assembler.inspect_pretty()       # Returns formatted summary
```

### ContextInspector

```python
ContextInspector(tokenizer="cl100k")
inspector.analyze(messages)  # Returns InspectionReport
inspector.diff(before, after)  # Returns InspectionDiff
inspector.trace(assembler)  # Returns BuildTrace
```

---

## Installation

```bash
# Basic install (uses approximate token counting)
pip install llm-contextkit

# With accurate OpenAI token counting
pip install llm-contextkit[tiktoken]

# Development install
pip install llm-contextkit[dev]
```

---

## Design Philosophy

- **Composable, not monolithic** — Pick the pieces you need, no forced framework adoption
- **Opinionated defaults, full override** — Sensible defaults, everything configurable
- **Model-agnostic** — Works with OpenAI, Anthropic, open-source models, any LLM
- **Observable by default** — Every operation is inspectable and debuggable
- **Library, not a service** — No infrastructure dependency, just `pip install`

---

## What ContextKit Does NOT Do

- **Retrieval / Embeddings** — Use your vector database (Pinecone, Weaviate, Qdrant)
- **LLM API calls** — We assemble the context; you send it however you want
- **Model-specific prompt tuning** — Too opinionated, varies by model
- **Authentication / Hosting** — Service territory, not library territory

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone and install for development
git clone https://github.com/llm-contextkit/llm-contextkit.git
cd contextkit
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=llm_contextkit --cov-report=term-missing
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
