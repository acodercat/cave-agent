from abc import ABC, abstractmethod
from typing import Any, List, Dict
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
    finish_reason: str | None = None


class StreamResponse(ABC):
    """Async iterator over streamed tokens with token usage available after exhaustion."""

    def __init__(self):
        self.usage = TokenUsage()
        self.finish_reason: str | None = None

    def _process_stream_chunk(self, chunk: Any) -> str | None:
        """Extract content text from a streaming chunk, updating finish_reason.

        Returns the content string if present, None otherwise.
        """
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if choice.finish_reason:
                self.finish_reason = choice.finish_reason
                return None
            if choice.delta.content:
                return choice.delta.content
        return None

    @abstractmethod
    def __aiter__(self):
        ...

    @abstractmethod
    async def __anext__(self) -> str:
        ...


class Model(ABC):
    """
    Abstract base class for language model engines.
    Defines interface for interacting with different LLM providers.
    """

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsage:
        """Extract token usage from an LLM API response."""
        if hasattr(response, "usage") and response.usage:
            return TokenUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
            )
        return TokenUsage()

    @staticmethod
    def _extract_response(response: Any) -> tuple[str, str | None]:
        """Extract content and finish_reason from an OpenAI-style response."""
        content = ""
        finish_reason = None
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
        return content, finish_reason

    @abstractmethod
    async def call(self, messages: List[Dict[str, str]]) -> ModelResponse:
        """Generate response from message history asynchronously.

        Returns:
            ModelResponse containing content and token usage.
        """
        pass

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]]) -> StreamResponse:
        """Stream response tokens from message history asynchronously.

        Returns:
            A StreamResponse that yields tokens and provides usage after exhaustion.
        """
        pass
