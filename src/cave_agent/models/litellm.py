from typing import List, Dict, Optional, Any, AsyncIterator

from .base import Model, ModelResponse, TokenUsage


class LiteLLMModel(Model):
    """
    LiteLLM model implementation that provides a unified interface to hundreds of LLM providers.

    LiteLLM is a library that standardizes the API for different LLM providers, allowing you to
    easily switch between OpenAI, Anthropic, Google, Azure, and many other providers with a
    consistent interface. This model acts as a gateway to access any LLM supported by LiteLLM.

    See https://www.litellm.ai/ for more information about supported providers and models.
    """

    def __init__(
            self,
            model_id: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            **kwargs
        ):
        """Initialize LiteLLM model.

        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL
            **kwargs: Additional parameters to pass to the API
        """
        try:
            import litellm
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'cave_agent[litellm]'`"
            )
        self.kwargs = kwargs
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key

    def _prepare_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare parameters for API call"""
        params = {
            "model": self.model_id,
            "api_base": self.base_url,
            "api_key": self.api_key,
            "messages": messages,
            **self.kwargs,
        }

        return params

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from LiteLLM response."""
        if hasattr(response, 'usage') and response.usage:
            return TokenUsage(
                prompt_tokens=getattr(response.usage, 'prompt_tokens', 0) or 0,
                completion_tokens=getattr(response.usage, 'completion_tokens', 0) or 0,
                total_tokens=getattr(response.usage, 'total_tokens', 0) or 0
            )
        return TokenUsage()

    async def call(self, messages: List[Dict[str, str]]) -> ModelResponse:
        """Generate response."""
        import litellm
        response = await litellm.acompletion(**self._prepare_params(messages), stream=False)

        content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content or ""

        return ModelResponse(
            content=content,
            token_usage=self._extract_token_usage(response)
        )

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens"""
        import litellm
        response = await litellm.acompletion(**self._prepare_params(messages), stream=True)

        async for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
