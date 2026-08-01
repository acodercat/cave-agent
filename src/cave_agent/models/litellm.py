from collections.abc import AsyncIterator
from typing import Any, cast

from .base import Model, ModelResponse, StreamResponse


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
        base_url: str | None = None,
        api_key: str | None = None,
        max_output_tokens: int | None = None,
        filters_asynchronously: bool = False,
        **kwargs,
    ):
        """Initialize LiteLLM model.

        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL
            max_output_tokens: Completion-token ceiling. Declared so the
                compactor can size its threshold, and sent on the wire so the
                declaration matches the real request.
            filters_asynchronously: Whether this endpoint may deliver content
                before its filter verdict. Such streams are drained before code
                is executed.
            **kwargs: Additional parameters to pass to the API
        """
        try:
            import litellm
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'cave_agent[litellm]'`"
            ) from None
        self._litellm = litellm
        self.kwargs = kwargs
        self.model_id = model_id
        self.filters_asynchronously = filters_asynchronously
        self.base_url = base_url
        self.api_key = api_key
        wire_cap = kwargs.get("max_tokens")
        if max_output_tokens is not None and wire_cap is not None and max_output_tokens != wire_cap:
            raise ValueError("max_output_tokens must match max_tokens when both are provided")
        if max_output_tokens is not None:
            # Assigned rather than `setdefault`: an explicit `max_tokens=None`
            # in kwargs is a *present* key, so setdefault kept the None and
            # dropped the declaration — leaving `max_output_tokens` None, and
            # the compactor sizing its threshold against a fallback.
            self.kwargs["max_tokens"] = max_output_tokens
        self.max_output_tokens = self.kwargs.get("max_tokens")
        self._stream_options_supported: bool | None = None

    def _supports_stream_options(self) -> bool:
        """Whether this model's provider accepts ``stream_options``.

        Support varies by provider and moves between LiteLLM releases, and
        LiteLLM **raises** on an unsupported parameter unless
        ``litellm.drop_params`` is on (it is off by default). So the capability
        is probed rather than assumed — asking unconditionally would break
        exactly the providers LiteLLM exists to reach.

        Deliberately no list of which providers accept it: the one that used to
        be here named Bedrock as rejecting and Vertex as accepting, and the
        installed LiteLLM reported the opposite for both. A comment that
        restates what the probe already answers is a second source of truth
        with no way to stay right.

        Providers that decline it may report zero streamed token usage. The
        agent estimates such turns rather than leaving them at zero, so the
        budgets keep moving — on an estimate rather than the authoritative
        count. Cached because it is a pure function of the model id.
        """
        if self._stream_options_supported is None:
            try:
                supported = (
                    self._litellm.get_supported_openai_params(
                        model=self.model_id,
                        custom_llm_provider=self.kwargs.get("custom_llm_provider"),
                    )
                    or []
                )
                self._stream_options_supported = "stream_options" in supported
            except Exception:
                # Unknown model or a LiteLLM version without the helper: assume
                # unsupported, since a wrong "yes" fails the request outright
                # while a wrong "no" only costs token accounting.
                self._stream_options_supported = False
        return self._stream_options_supported

    def _prepare_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Prepare parameters for API call."""
        return {
            "model": self.model_id,
            "api_base": self.base_url,
            "api_key": self.api_key,
            "messages": messages,
            **self.kwargs,
        }

    async def _complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        """Generate response."""
        params = self._prepare_params(messages)
        params["stream"] = False
        # ``stream_options`` is meaningless without a stream, and providers
        # reject the pair with a 400. A caller may legitimately pass their own
        # — ``_open_stream`` invites it with ``setdefault`` — and ``call()`` is
        # the compactor's only path, so leaving it here took summarization down
        # while streaming kept working.
        params.pop("stream_options", None)
        response = await self._litellm.acompletion(**params)

        return self._build_response(response)

    def stream(self, messages: list[dict[str, str]]) -> StreamResponse:
        """Stream response tokens."""
        return _LiteLLMStreamResponse(self, messages)


class _LiteLLMStreamResponse(StreamResponse):
    """LiteLLM streaming response — iteration, usage capture, connect-retry, and
    close are all handled by :class:`StreamResponse`; this only opens the stream."""

    def __init__(self, model: LiteLLMModel, messages: list[dict[str, str]]):
        super().__init__()
        self._model = model
        self._messages = messages

    async def _open_stream(self) -> AsyncIterator[Any]:
        params = self._model._prepare_params(self._messages)
        if self._model._supports_stream_options():
            # Ask for the terminal usage chunk. Without it a streamed run
            # reports zero tokens, which silently disables the cumulative token
            # budgets and leaves compaction estimating from the character
            # heuristic alone instead of the authoritative API count.
            # ``setdefault`` so a caller who passed their own wins.
            params.setdefault("stream_options", {"include_usage": True})
        # In params, not alongside them: ``stream`` is a legitimate provider
        # kwarg, and passing it twice fails the call before it leaves here.
        params["stream"] = True
        return cast(
            AsyncIterator[Any],
            await self._model._litellm.acompletion(**params),
        )
