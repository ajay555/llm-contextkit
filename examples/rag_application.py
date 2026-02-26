#!/usr/bin/env python3
"""RAG (Retrieval-Augmented Generation) application example using ContextKit.

This example demonstrates how to use ContextKit to manage context for a
knowledge assistant that answers questions based on retrieved documents.

Usage:
    python rag_application.py

Requirements:
    pip install contextkit
    pip install anthropic  # Optional, for actual API calls
"""

from llm_contextkit import ContextAssembler, ContextInspector, TokenBudget
from llm_contextkit.layers import (
    HistoryLayer,
    RetrievedLayer,
    SystemLayer,
    UserContextLayer,
)

# Simulated retrieved documents (in practice, these come from a vector database)
retrieved_chunks = [
    {
        "text": """Enterprise customers are entitled to a 60-day refund window for all
subscription purchases. Refund requests must be submitted through the admin
portal or by contacting enterprise support. Partial refunds are calculated
on a pro-rata basis from the date of the refund request.""",
        "source": "refund-policy-v4.pdf",
        "score": 0.94,
    },
    {
        "text": """To request a refund, enterprise administrators should navigate to
Settings > Billing > Refund Requests. Click 'New Request' and fill out the
required information including reason for refund and preferred refund method.
Requests are typically processed within 5-7 business days.""",
        "source": "admin-guide.md",
        "score": 0.89,
    },
    {
        "text": """Standard tier customers have a 30-day refund window. After 30 days,
refunds are only available in cases of service outages exceeding 24 hours
or documented product defects. Contact support@acme.com for assistance.""",
        "source": "refund-policy-v4.pdf",
        "score": 0.72,
    },
    {
        "text": """Refund processing times vary by payment method: Credit cards take
5-7 business days, bank transfers take 7-10 business days, and PayPal
refunds are typically instant but may take up to 24 hours.""",
        "source": "billing-faq.md",
        "score": 0.68,
    },
]

# User context (from your user database/session)
user_context = {
    "name": "Jane Smith",
    "email": "jane.smith@enterprise.com",
    "role": "Enterprise Admin",
    "account_tier": "Enterprise",
    "account_id": "ENT-12345",
    "region": "US-West",
    "subscription_start": "2024-01-15",
}

# Conversation history
conversation_history = [
    {"role": "user", "content": "Hi, I need to request a refund for our team subscription."},
    {"role": "assistant", "content": "I'd be happy to help you with a refund request. As an Enterprise customer, you have a 60-day refund window. Could you tell me more about your situation?"},
    {"role": "user", "content": "We purchased 50 seats but only ended up using 30. Can we get a refund for the unused seats?"},
]


def main():
    # Create a token budget for Claude 3 (200k context, but we'll use less)
    budget = TokenBudget(
        total=16000,
        tokenizer="approximate",
        reserve_for_output=2000,
    )

    # Allocate budgets with priorities
    budget.allocate("system", 800, priority=10)
    budget.allocate("user_context", 400, priority=8)
    budget.allocate("retrieved_docs", 6000, priority=6)
    budget.allocate("history", 6000, priority=5)

    # Create the assembler
    assembler = ContextAssembler(budget=budget)

    # Add system layer
    assembler.add_layer(
        SystemLayer(
            instructions="""You are a knowledgeable customer support assistant for Acme SaaS.

Your primary role is to answer questions based ONLY on the provided documents.
Follow these guidelines:
1. Base your answers strictly on the retrieved documents
2. If the documents don't contain enough information, say so clearly
3. Cite your sources by mentioning the document name
4. Consider the customer's account tier when providing information
5. Be helpful and professional

When discussing refunds or billing, always verify the customer's eligibility
based on their account tier and subscription details."""
        )
    )

    # Add user context layer
    assembler.add_layer(
        UserContextLayer(context=user_context)
    )

    # Add retrieved documents layer
    assembler.add_layer(
        RetrievedLayer(
            chunks=retrieved_chunks,
            max_chunks=5,
            include_metadata=True,
        )
    )

    # Add conversation history
    assembler.add_layer(
        HistoryLayer(
            messages=conversation_history,
            strategy="sliding_window",
            strategy_config={"max_turns": 10},
        )
    )

    # Build for Anthropic API format
    payload = assembler.build_for_anthropic()

    # Print the result
    print("=" * 60)
    print("CONTEXT BUILT FOR ANTHROPIC API")
    print("=" * 60)
    print()
    print("[SYSTEM]")
    print(payload["system"][:500] + "..." if len(payload["system"]) > 500 else payload["system"])
    print()
    print("[MESSAGES]")
    for msg in payload["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  {role}: {content}")
    print()

    # Print inspection summary
    print("=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    print(assembler.inspect_pretty())

    # Use the standalone inspector to analyze the context
    print()
    print("=" * 60)
    print("DETAILED INSPECTION")
    print("=" * 60)
    inspector = ContextInspector(tokenizer="approximate")

    # Convert to messages format for inspection
    messages_for_inspection = [{"role": "system", "content": payload["system"]}]
    messages_for_inspection.extend(payload["messages"])

    report = inspector.analyze(messages_for_inspection)
    print(report.pretty())

    # Example: Send to Anthropic (commented out - requires API key)
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-3-opus-20240229",
    #     max_tokens=budget.reserve_for_output,
    #     system=payload["system"],
    #     messages=payload["messages"],
    # )
    # print(response.content[0].text)


if __name__ == "__main__":
    main()
