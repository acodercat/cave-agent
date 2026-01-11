from typing import List, Dict, Optional, Any, AsyncIterator

from .base import Model, ModelResponse, TokenUsage


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

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from OpenAI response."""
        if hasattr(response, 'usage') and response.usage:
            return TokenUsage(
                prompt_tokens=getattr(response.usage, 'prompt_tokens', 0) or 0,
                completion_tokens=getattr(response.usage, 'completion_tokens', 0) or 0,
                total_tokens=getattr(response.usage, 'total_tokens', 0) or 0
            )
        return TokenUsage()

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

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens using OpenAI API."""
        response = await self.client.chat.completions.create(
            **self._prepare_params(messages),
            stream=True
        )

        async for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
