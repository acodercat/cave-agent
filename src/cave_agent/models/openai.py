from typing import List, Dict, Optional, Any

from .base import Model, ModelResponse, StreamResponse
from .retry import with_retry_typed


class OpenAIModel(Model):
    """
    OpenAI-compatible LLM engine implementation.
    Supports OpenAI API and compatible endpoints.
    """

    def __init__(
            self,
            model_id: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            organization: Optional[str] = None,
            project: Optional[str] = None,
            **kwargs
        ):
        """Initialize OpenAI model.

        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL
            organization: Optional organization ID
            project: Optional project ID
            **kwargs: Additional parameters to pass to the OpenAI API
        """
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIModel: `pip install 'cave_agent[openai]'`"
            )

        self.kwargs = kwargs
        self.model_id = model_id
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            project=project,
        )

    def _prepare_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API call."""
        return {
            "model": self.model_id,
            "messages": messages,
            **self.kwargs,
        }

    async def call(self, messages: List[Dict[str, str]]) -> ModelResponse:
        """Generate response using OpenAI API asynchronously."""
        params = self._prepare_params(messages)
        response = await with_retry_typed(
            lambda: self.client.chat.completions.create(**params, stream=False)
        )

        content, finish_reason = self._extract_response(response)

        return ModelResponse(
            content=content,
            token_usage=self._extract_token_usage(response),
            finish_reason=finish_reason,
            thinking=self._extract_thinking(response),
        )

    def stream(self, messages: List[Dict[str, str]]) -> StreamResponse:
        """Stream response tokens using OpenAI API."""
        return _OpenAIStreamResponse(self, messages)

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI client's connection pool."""
        await self.client.close()


class _OpenAIStreamResponse(StreamResponse):
    """OpenAI streaming response — iteration, usage capture, connect-retry, and
    close are all handled by :class:`StreamResponse`; this only opens the stream."""

    def __init__(self, model: OpenAIModel, messages: List[Dict[str, str]]):
        super().__init__()
        self._model = model
        self._messages = messages

    async def _open_stream(self):
        params = self._model._prepare_params(self._messages)
        return await self._model.client.chat.completions.create(
            **params, stream=True, stream_options={"include_usage": True},
        )
