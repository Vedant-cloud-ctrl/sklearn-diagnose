"""
LLM integration for sklearn-diagnose using LangChain.

This module provides LLM-powered AI agents for:
1. Hypothesis generation agent (detecting failure modes)
2. Recommendation generation agent (actionable suggestions)
3. Summary generation agent (human-readable summaries)

Supports:
- OpenAI (via langchain-openai)
- Anthropic (via langchain-anthropic)
- OpenRouter (via langchain-openai with custom base_url)
- Groq (via langchain-groq)

IMPORTANT: You must call setup_llm() before using diagnose().

Example:
    >>> from sklearn_diagnose import setup_llm, diagnose
    >>> setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")
    >>> report = diagnose(model, datasets, task="classification")
"""

from .client import (
    # Base classes
    LLMClient,
    LangChainClient,
    # Provider-specific clients
    OpenAIClient,
    AnthropicClient,
    OpenRouterClient,
    GroqChatClient,
    # Configuration
    setup_llm,
    _set_global_client,
    _get_global_client,
    # Generation functions
    generate_llm_hypotheses,
    generate_llm_recommendations,
    generate_llm_summary,
)

__all__ = [
    # Base classes
    "LLMClient",
    "LangChainClient",
    # Provider-specific clients
    "OpenAIClient",
    "AnthropicClient",
    "OpenRouterClient",
    "GroqChatClient",
    # Configuration
    "setup_llm",
    # Generation functions
    "generate_llm_hypotheses",
    "generate_llm_recommendations",
    "generate_llm_summary",
]
