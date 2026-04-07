from typing import List, Dict, Optional, Any

from .base import Model, ModelResponse, StreamResponse, TokenUsage


class OpenAIServerModel(Model):
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
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'cave_agent[openai]'`"
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
        params = {
            "model": self.model_id,
            "messages": messages,
            **self.kwargs,
        }

        return params

    async def call(self, messages: List[Dict[str, str]]) -> ModelResponse:
        """Generate response using OpenAI API asynchronously."""
        response = await self.client.chat.completions.create(
            **self._prepare_params(messages),
            stream=False
        )

        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content or ""

        return ModelResponse(
            content=content,
            token_usage=self._extract_token_usage(response)
        )

    def stream(self, messages: List[Dict[str, str]]) -> StreamResponse:
        """Stream response tokens using OpenAI API."""
        return _OpenAIStreamResponse(self, messages)


class _OpenAIStreamResponse(StreamResponse):
    """Async iterator over OpenAI streaming chunks with usage tracking."""

    def __init__(self, model: OpenAIServerModel, messages: List[Dict[str, str]]):
        super().__init__()
        self._model = model
        self._messages = messages
        self._iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._iterator is None:
            response = await self._model.client.chat.completions.create(
                **self._model._prepare_params(self._messages),
                stream=True,
                stream_options={"include_usage": True},
            )
            self._iterator = response.__aiter__()

        while True:
            chunk = await self._iterator.__anext__()

            if hasattr(chunk, "usage") and chunk.usage:
                self.usage = self._model._extract_token_usage(chunk)
                continue

            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    return delta.content
