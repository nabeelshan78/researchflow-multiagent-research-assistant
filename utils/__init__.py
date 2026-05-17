"""
Utility package for the Multi-Agent Business Research Assistant.

Provides:
    - LLM client factory functions (dual-model: fast & reasoning)
    - Search tool instantiation
    - Custom exception hierarchy for structured error handling
"""

from utils.errors import LLMAPIError, SearchToolError, StateValidationError
from utils.llm import get_llm, get_search_tool

__all__: list[str] = [
    "LLMAPIError",
    "SearchToolError",
    "StateValidationError",
    "get_llm",
    "get_search_tool",
]
