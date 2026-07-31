"""Provider-agnostic error classification for the model layer.

Concrete ``Model`` implementations translate their SDK's exceptions into
these types so the agent loop can ``except`` on semantic error types without
knowing the provider.

Classification is **total**: :func:`classify_provider_error` maps anything to a
:class:`ModelError`, and :func:`raise_model_error` is the single way an
exception leaves this layer. Nothing untyped reaches the agent.
"""

from typing import NoReturn


class ModelError(Exception):
    """Base class for all provider-agnostic model-layer errors."""


class PromptTooLongError(ModelError):
    """The request exceeded the model's context window.

    Providers raise this on HTTP 413, OpenAI's ``context_length_exceeded``
    code, or a matching message (see :func:`is_context_length_exceeded`).
    Recovered by compaction, never by retry.
    """


class ProviderBillingError(ModelError):
    """The endpoint's account is out of credit or paid quota.

    Deterministic for this endpoint until a human tops up, so it is never
    retried (a top-up, not time, is what fixes it).
    """


class ProviderError(ModelError):
    """A provider failure with no more specific classification.

    The terminal arm of :func:`classify_provider_error`: a dropped connection,
    a malformed response body, an SDK exception this release has never seen.
    Typed so it ends the run with ``StopReason.MODEL_ERROR`` instead of
    unwinding out of the agent loop past the terminal event every consumer
    relies on. The original exception is always the ``__cause__``.
    """


class StreamStalledError(ModelError, TimeoutError):
    """A streaming response produced no chunk within ``stream_idle_timeout``.

    Typed as a :class:`ModelError` so the agent loop ends the run with
    ``StopReason.MODEL_ERROR`` instead of letting a bare ``TimeoutError``
    escape. A caller streaming to a browser needs a terminal event it can
    forward, not an exception unwinding through its response middleware.

    Also a :class:`TimeoutError`, so code that already catches that keeps
    working — the classification is added, not swapped.
    """


# Substrings that mark a context-length / prompt-too-long overflow. Kept
# SPECIFIC on purpose: a bare "max_tokens" would swallow a "max_tokens must
# be positive" validation 400, and a false positive here triggers
# (destructive) compaction rather than surfacing the real error.
_CONTEXT_LENGTH_PHRASES = (
    # OpenAI / Moonshot / generic OpenAI-compatible
    "context length",
    "context window",
    "maximum number of tokens",
    # vLLM / SGLang
    "maximum allowed length",
    "max_model_len",
    "maximum model length",
    "prompt length",
    # llama.cpp
    "n_ctx_slot",
    "slot context",
    # Ollama
    "truncating input",
    # Anthropic (native or behind an OpenAI-compat proxy)
    "prompt is too long",
    "input is too long",
    # Gemini
    "input token count",
    # AWS Bedrock
    "number of input tokens",
    # Chinese-language providers (DashScope / Zhipu GLM)
    "上下文长度",
    "超过最大长度",
)


def is_context_length_exceeded(error: Exception) -> bool:
    """Detect "the prompt exceeds the context window" across providers.

    HTTP 413 and OpenAI's ``code='context_length_exceeded'`` are structured
    signals; everyone else — and the OpenAI-compatible serving stacks behind
    one base URL — only signal it in the message, so fall back to phrase
    matching.
    """
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status == 413:
        return True
    if getattr(error, "code", None) == "context_length_exceeded":
        return True
    message = str(error).lower()
    return any(phrase in message for phrase in _CONTEXT_LENGTH_PHRASES)


# Signals that an endpoint is out of credit/quota. HTTP 402 is the structured
# form (OpenRouter); most providers only put the signal in the message body —
# OpenAI's ``insufficient_quota`` rides a 429, Anthropic's "credit balance is
# too low" a 400. Phrases stay specific so a bare "quota"/"exhausted" doesn't
# swallow transient rate limits (which must retry, not fail).
_BILLING_PHRASES = (
    # OpenAI / Azure
    "insufficient_quota",
    "exceeded your current quota",
    "billing hard limit",
    # OpenRouter
    "insufficient credits",
    "credits exhausted",
    "credits have been exhausted",
    "no usable credits",
    "top up your credits",
    "key limit exceeded",
    "spending limit",
    # Anthropic
    "credit balance",
    "out of extra usage",
    # DashScope / generic balance wording
    "insufficient balance",
    "insufficient_balance",
    "balance is not enough",
    "balance_depleted",
    "arrearage",
    # Cross-provider generic
    "payment required",
    "out of funds",
    "run out of funds",
)


def is_billing_exhausted(error: Exception) -> bool:
    """Detect "this endpoint's account is out of money" across providers.

    Called from the retry predicate to skip retrying it, and from provider
    translation arms to route it to :class:`ProviderBillingError`.
    """
    status = (
        getattr(error, "status_code", None)
        or getattr(error, "status", None)
        or getattr(error, "code", None)
    )
    if status == 402:
        return True
    message = str(error).lower()
    return any(phrase in message for phrase in _BILLING_PHRASES)


def classify_provider_error(error: Exception) -> ModelError:
    """Map a provider SDK exception onto the typed hierarchy.

    **Total** — every input yields a :class:`ModelError`, falling back to
    :class:`ProviderError`. That totality is the point: while this returned
    unclassified errors unchanged, every call site needed its own
    ``if typed is error`` fallback, and each of those fallbacks was a hole
    through which a raw ``ConnectionError`` escaped the agent loop.

    Already-typed errors pass through untouched (idempotent), so wrapping a
    call twice cannot double-translate.
    """
    if isinstance(error, ModelError):
        return error
    if is_context_length_exceeded(error):
        return PromptTooLongError(str(error))
    if is_billing_exhausted(error):
        return ProviderBillingError(str(error))
    return ProviderError(f"{type(error).__name__}: {error}")


def raise_model_error(error: Exception) -> NoReturn:
    """Re-raise *error* as a typed :class:`ModelError`, preserving the cause.

    The one way out of the model layer. Exists so no boundary re-implements
    the classify-and-chain dance, and so adding a boundary is a one-liner that
    cannot get the ``raise … from …`` wrong.
    """
    typed = classify_provider_error(error)
    if typed is error:
        raise error
    raise typed from error
