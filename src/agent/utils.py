"""
Shared utility functions.
"""

from __future__ import annotations
import os
from langchain_core.language_models import BaseChatModel


def get_llm(model_name: str) -> BaseChatModel:
    """
    LLM factory. Returns the appropriate LangChain chat model instance.

    Supports:
      - DeepSeek: model names starting with "deepseek-"
        Uses the official langchain-deepseek package (ChatDeepSeek).
        Requires: pip install langchain-deepseek
        Env var:  DEEPSEEK_API_KEY

      - Gemini: model names starting with "gemini-"
        Uses langchain-google-genai (ChatGoogleGenerativeAI).
        Requires: pip install langchain-google-genai
        Env var:  GEMINI_API_KEY

    Model name reference:
      deepseek-chat      → DeepSeek-V3.2, non-thinking mode (supports tool calling + structured output)
      deepseek-reasoner  → DeepSeek-V3.2, thinking mode (NOT supported for tool calling)
      gemini-2.0-flash   → Gemini 2.0 Flash
    """
    if model_name.startswith("deepseek"):
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=model_name,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            temperature=0.0,
            timeout=60.0,
            max_retries=2,
        )
    elif model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.0,
            timeout=60.0,
            max_retries=2,
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Expected a name starting with 'deepseek-' or 'gemini-'."
        )