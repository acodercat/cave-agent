"""Provider-agnostic error classification for the model layer.

Concrete ``Model`` implementations translate their SDK's exceptions into
these types so the agent loop can ``except`` on semantic error types without
knowing the provider.
"""


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

    OpenAI sets a structured ``code='context_length_exceeded'`` (checked
    first); everyone else — and the OpenAI-compatible serving stacks behind
    one base URL — only signal it in the message, so fall back to phrase
    matching.
    """
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


def classify_provider_error(error: Exception) -> Exception:
    """Map a raw provider SDK exception to a typed :class:`ModelError` when it
    matches a known category, else return it unchanged.

    Already-typed :class:`ModelError`\\ s pass through untouched (idempotent),
    so wrapping a call twice can't double-translate.
    """
    if isinstance(error, ModelError):
        return error
    if is_context_length_exceeded(error):
        return PromptTooLongError(str(error))
    if is_billing_exhausted(error):
        return ProviderBillingError(str(error))
    return error
