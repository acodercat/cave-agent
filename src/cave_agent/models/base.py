from abc import ABC, abstractmethod
from typing import List, Dict, AsyncIterator
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage objects together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class ModelResponse:
    """Response from an LLM model call including token usage."""
    content: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)


class Model(ABC):
    """
    Abstract base class for language model engines.
    Defines interface for interacting with different LLM providers.
    """

    @abstractmethod
    async def call(self, messages: List[Dict[str, str]]) -> ModelResponse:
        """Generate response from message history asynchronously.

        Returns:
            ModelResponse containing content and token usage.
        """
        pass

    @abstractmethod
    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens from message history asynchronously."""
        pass
