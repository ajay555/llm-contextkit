#!/usr/bin/env python3
"""Agentic workflow example using ContextKit.

This example demonstrates how to use ContextKit to manage context for a
multi-step research agent that uses tools and maintains state across
multiple iterations.

Usage:
    python agentic_workflow.py

Requirements:
    pip install contextkit
    pip install openai  # Optional, for actual API calls
"""

from llm_contextkit import ContextAssembler, ContextInspector, TokenBudget
from llm_contextkit.layers import (
    HistoryLayer,
    RetrievedLayer,
    SystemLayer,
    ToolResultsLayer,
)

# Simulated tool results from previous agent steps
tool_results = [
    {
        "tool_name": "web_search",
        "input": {"query": "ACME Corp Q3 2024 revenue earnings"},
        "output": """ACME Corp Q3 2024 Results:
- Revenue: $2.3 billion (up 15% YoY)
- Net income: $340 million
- EPS: $1.42 (beat estimates of $1.35)
- Cloud segment grew 28% to $890M
- Guidance raised for full year""",
        "status": "success",
    },
    {
        "tool_name": "database_query",
        "input": {"sql": "SELECT * FROM company_financials WHERE ticker='ACME' ORDER BY quarter DESC LIMIT 4"},
        "output": """Q3 2024: Revenue $2.3B, Net Income $340M
Q2 2024: Revenue $2.1B, Net Income $298M
Q1 2024: Revenue $1.9B, Net Income $267M
Q4 2023: Revenue $2.0B, Net Income $285M""",
        "status": "success",
    },
    {
        "tool_name": "web_search",
        "input": {"query": "ACME Corp competitor analysis technology sector"},
        "output": """Top Competitors:
1. TechGiant Inc - Market cap $50B, similar cloud offerings
2. DataSystems Corp - Stronger in enterprise, weaker in SMB
3. CloudFirst Ltd - Growing fast in Asia-Pacific region
Industry analysts rate ACME as #2 in market share.""",
        "status": "success",
    },
    {
        "tool_name": "calculator",
        "input": {"expression": "(2.3 - 2.0) / 2.0 * 100"},
        "output": "15.0 (representing 15% revenue growth Q3 2024 vs Q4 2023)",
        "status": "success",
    },
]

# Retrieved context documents
retrieved_chunks = [
    {
        "text": """ACME Corp Investment Thesis:
Strong fundamentals with consistent revenue growth. Cloud segment is the
primary growth driver. Management has proven ability to execute. Main risks
include increased competition and potential economic slowdown affecting
enterprise IT spending.""",
        "source": "internal-research-acme.pdf",
        "score": 0.91,
    },
    {
        "text": """Technology Sector Outlook Q4 2024:
Enterprise software spending expected to remain robust despite macro concerns.
Cloud adoption continues to accelerate. AI/ML integration becoming a key
differentiator. Companies with strong recurring revenue models are favored.""",
        "source": "sector-outlook-q4.pdf",
        "score": 0.84,
    },
]

# Agent conversation history
agent_history = [
    {"role": "user", "content": "Research ACME Corp and provide an investment analysis with a recommendation."},
    {"role": "assistant", "content": "I'll research ACME Corp comprehensively. Let me start by gathering their latest financial data and market position."},
    {"role": "user", "content": "Focus especially on their growth trajectory and competitive positioning."},
    {"role": "assistant", "content": "I've gathered the Q3 2024 earnings data and competitor information. Now let me analyze the historical trends and compile the analysis."},
]


def main():
    # Create a token budget for a capable model (16k context)
    budget = TokenBudget(
        total=16000,
        tokenizer="approximate",
        reserve_for_output=2000,
    )

    # Allocate budgets with priorities
    # Tool results get high priority - agent needs them to reason
    budget.allocate("system", 1000, priority=10)
    budget.allocate("tool_results", 6000, priority=8)
    budget.allocate("retrieved_docs", 3000, priority=6)
    budget.allocate("history", 3000, priority=5)

    # Create the assembler
    assembler = ContextAssembler(budget=budget)

    # Add system layer with agent instructions
    assembler.add_layer(
        SystemLayer(
            instructions="""You are a financial research agent with access to various tools.

Your capabilities:
- web_search: Search the internet for current information
- database_query: Query internal financial databases
- calculator: Perform numerical calculations

Your workflow:
1. Analyze the user's research request
2. Use available tools to gather relevant data
3. Synthesize findings into actionable insights
4. Provide clear recommendations with supporting evidence

Guidelines:
- Always cite your sources (tool outputs, documents)
- Show your reasoning process
- Quantify claims with data when possible
- Acknowledge limitations or data gaps
- Provide balanced analysis including risks

Current task: Provide investment analysis based on gathered data."""
        )
    )

    # Add tool results layer with full inputs for debugging
    assembler.add_layer(
        ToolResultsLayer(
            results=tool_results,
            include_inputs=True,  # Include inputs for transparency
        )
    )

    # Add retrieved research documents
    assembler.add_layer(
        RetrievedLayer(
            chunks=retrieved_chunks,
            max_chunks=5,
            include_metadata=True,
        )
    )

    # Add agent conversation history
    assembler.add_layer(
        HistoryLayer(
            messages=agent_history,
            strategy="sliding_window",
            strategy_config={"max_turns": 20},
        )
    )

    # Build for OpenAI API format
    messages = assembler.build_for_openai()

    # Print the result
    print("=" * 70)
    print("AGENTIC WORKFLOW - CONTEXT FOR NEXT REASONING STEP")
    print("=" * 70)
    print()

    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        print(f"[{role}]")
        if len(content) > 300:
            print(content[:300] + f"... ({len(content)} chars total)")
        else:
            print(content)
        print()

    # Print build summary
    print("=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)
    print(assembler.inspect_pretty())

    # Demonstrate context inspection
    print()
    print("=" * 70)
    print("CONTEXT ANALYSIS")
    print("=" * 70)
    inspector = ContextInspector(tokenizer="approximate")
    report = inspector.analyze(messages)
    print(report.pretty())

    # Demonstrate what happens when we add more tool results
    print()
    print("=" * 70)
    print("SIMULATING CONTEXT GROWTH")
    print("=" * 70)

    # Simulate adding another tool result (new agent step)
    new_tool_result = {
        "tool_name": "web_search",
        "input": {"query": "ACME Corp analyst ratings price targets"},
        "output": """Analyst Ratings for ACME Corp:
- Goldman Sachs: BUY, PT $145
- Morgan Stanley: OVERWEIGHT, PT $140
- JP Morgan: NEUTRAL, PT $125
- Average PT: $137 (current price: $128)
- 12 analysts covering, 8 Buy, 3 Hold, 1 Sell""",
        "status": "success",
    }

    # Create new assembler with updated tool results
    extended_results = tool_results + [new_tool_result]

    budget2 = TokenBudget(total=16000, tokenizer="approximate", reserve_for_output=2000)
    budget2.allocate("system", 1000, priority=10)
    budget2.allocate("tool_results", 6000, priority=8)
    budget2.allocate("retrieved_docs", 3000, priority=6)
    budget2.allocate("history", 3000, priority=5)

    assembler2 = ContextAssembler(budget=budget2)
    assembler2.add_layer(SystemLayer(instructions="Research agent system prompt..."))
    assembler2.add_layer(ToolResultsLayer(results=extended_results, include_inputs=True))
    assembler2.add_layer(RetrievedLayer(chunks=retrieved_chunks))
    assembler2.add_layer(HistoryLayer(messages=agent_history))

    messages2 = assembler2.build_for_openai()

    # Compare before and after
    diff = inspector.diff(messages, messages2)
    print(diff.pretty())

    # Example: Send to OpenAI (commented out - requires API key)
    # import openai
    # response = openai.chat.completions.create(
    #     model="gpt-4",
    #     messages=messages,
    #     max_tokens=budget.reserve_for_output,
    # )
    # print("\n[AGENT RESPONSE]")
    # print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
