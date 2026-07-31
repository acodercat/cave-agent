from typing import Any

from .base import Model, ModelResponse, StreamResponse


class OpenAIModel(Model):
    """
    OpenAI-compatible LLM engine implementation.
    Supports OpenAI API and compatible endpoints.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        max_output_tokens: int | None = None,
        filters_asynchronously: bool = False,
        **kwargs,
    ):
        """Initialize OpenAI model.

        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL
            organization: Optional organization ID
            project: Optional project ID
            max_output_tokens: Completion-token ceiling, declared so the
                compactor can size its threshold. Not sent on the wire — pass
                ``max_tokens`` or ``max_completion_tokens`` through kwargs to
                actually cap generation, whichever your model accepts.
            filters_asynchronously: Whether this endpoint may deliver content
                before its filter verdict. Such streams are drained before code
                is executed.
            **kwargs: Additional parameters to pass to the OpenAI API
        """
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIModel: `pip install 'cave_agent[openai]'`"
            ) from None

        self.kwargs = kwargs
        self.model_id = model_id
        self.filters_asynchronously = filters_asynchronously
        wire_caps = [
            (name, kwargs[name])
            for name in ("max_completion_tokens", "max_tokens")
            if kwargs.get(name) is not None
        ]
        if len(wire_caps) > 1:
            raise ValueError("Pass only one of max_completion_tokens and max_tokens")
        wire_cap = wire_caps[0][1] if wire_caps else None
        if max_output_tokens is not None and wire_cap is not None and max_output_tokens != wire_cap:
            raise ValueError(
                "max_output_tokens must match the wire token cap when both are provided"
            )
        # A declaration for the compactor's threshold — deliberately NOT put on
        # the wire. OpenAI spells the cap two ways (``max_tokens`` for chat
        # models, ``max_completion_tokens`` for the reasoning ones, which reject
        # the former with a 400), and guessing from the model id is a rule that
        # rots with every release. To actually cap generation, pass whichever
        # your model accepts through ``kwargs``; either is picked up below so
        # the declaration still matches what is sent.
        self.max_output_tokens = wire_cap if wire_cap is not None else max_output_tokens
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            project=project,
        )

    def _prepare_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Prepare parameters for OpenAI API call."""
        return {
            "model": self.model_id,
            "messages": messages,
            **self.kwargs,
        }

    async def _complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        """Generate response using OpenAI API asynchronously."""
        params = self._prepare_params(messages)
        params["stream"] = False
        params.pop("stream_options", None)
        response = await self.client.chat.completions.create(**params)

        return self._build_response(response)

    def stream(self, messages: list[dict[str, str]]) -> StreamResponse:
        """Stream response tokens using OpenAI API."""
        return _OpenAIStreamResponse(self, messages)

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI client's connection pool."""
        await self.client.close()


class _OpenAIStreamResponse(StreamResponse):
    """OpenAI streaming response — iteration, usage capture, connect-retry, and
    close are all handled by :class:`StreamResponse`; this only opens the stream."""

    def __init__(self, model: OpenAIModel, messages: list[dict[str, str]]):
        super().__init__()
        self._model = model
        self._messages = messages

    async def _open_stream(self):
        params = self._model._prepare_params(self._messages)
        # Merged into params rather than passed alongside them: a caller who
        # sets either of these through kwargs — both are legitimate provider
        # arguments — supplied it twice, and the call died with "got multiple
        # values for keyword argument" before ever reaching the provider.
        # ``setdefault`` so their value wins, matching ``LiteLLMModel``.
        params["stream"] = True
        params.setdefault("stream_options", {"include_usage": True})
        return await self._model.client.chat.completions.create(**params)
