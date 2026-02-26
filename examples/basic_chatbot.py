#!/usr/bin/env python3
"""Basic chatbot example using ContextKit.

This example demonstrates how to use ContextKit to manage context for a
simple customer support chatbot with conversation history.

Usage:
    python basic_chatbot.py

Requirements:
    pip install contextkit
    pip install openai  # Optional, for actual API calls
"""

from llm_contextkit import ContextAssembler, TokenBudget
from llm_contextkit.layers import HistoryLayer, SystemLayer

# Simulated conversation history
conversation_history = [
    {"role": "user", "content": "Hi, I'm having trouble logging into my account."},
    {"role": "assistant", "content": "I'm sorry to hear that! I'd be happy to help you with your login issue. Can you tell me what error message you're seeing?"},
    {"role": "user", "content": "It says 'Invalid credentials' but I'm sure my password is correct."},
    {"role": "assistant", "content": "That's frustrating. Let's try a few things. First, have you tried resetting your password recently? Sometimes cached credentials can cause this issue."},
    {"role": "user", "content": "No, I haven't. Should I try that?"},
    {"role": "assistant", "content": "Yes, I'd recommend it. Go to the login page and click 'Forgot Password'. You'll receive an email with a reset link."},
    {"role": "user", "content": "Okay, I'll try that. Also, can you tell me about your refund policy?"},
]


def main():
    # Create a token budget for GPT-4 (8k context)
    budget = TokenBudget(
        total=8192,
        tokenizer="approximate",  # Use "cl100k" if tiktoken is installed
        reserve_for_output=1000,
    )

    # Allocate budgets for each layer
    budget.allocate("system", 500, priority=10)  # High priority - rarely truncated
    budget.allocate("history", 6500, priority=5)  # Lower priority - truncated first

    # Create the assembler
    assembler = ContextAssembler(budget=budget)

    # Add system layer with instructions and few-shot examples
    assembler.add_layer(
        SystemLayer(
            instructions="""You are a helpful customer support agent for Acme SaaS.

Your responsibilities:
- Help customers with account issues, billing questions, and product features
- Be friendly, professional, and empathetic
- If you don't know something, say so and offer to escalate
- Always verify customer identity before discussing account details""",
            few_shot_examples=[
                {
                    "input": "I can't log in",
                    "output": "I'd be happy to help with your login issue. Let me verify your account status. Could you please provide your email address?",
                },
                {
                    "input": "I want a refund",
                    "output": "I understand you'd like a refund. I can help with that. Could you tell me more about the reason for your request so I can ensure we resolve any issues?",
                },
            ],
        )
    )

    # Add conversation history with sliding window strategy
    assembler.add_layer(
        HistoryLayer(
            messages=conversation_history,
            strategy="sliding_window",
            strategy_config={"max_turns": 10},
        )
    )

    # Build for OpenAI API format
    messages = assembler.build_for_openai()

    # Print the result
    print("=" * 60)
    print("CONTEXT BUILT FOR OPENAI API")
    print("=" * 60)
    print()
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        print(f"[{role}]")
        print(content)
        print()

    # Print inspection summary
    print("=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    print(assembler.inspect_pretty())

    # Example: Send to OpenAI (commented out - requires API key)
    # import openai
    # response = openai.chat.completions.create(
    #     model="gpt-4",
    #     messages=messages,
    #     max_tokens=budget.reserve_for_output,
    # )
    # print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
