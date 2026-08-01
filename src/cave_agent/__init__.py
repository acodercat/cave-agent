"""CaveAgent — function-calling through code generation against a live runtime.

The public API is what this module re-exports. Rendering lives in
``cave_agent.renderers`` and is not imported here: the agent core has no
dependency on it, so a headless deployment never pulls it in.
"""

from typing import TYPE_CHECKING

from .agent import AgentResponse, CaveAgent
from .compaction import Compactor
from .events import (
    CodeEvent,
    Event,
    ExecutionResultEvent,
    ExecutionTimeoutEvent,
    FinalResponseEvent,
    SecurityErrorEvent,
    StatusEvent,
    StatusType,
    StoppedEvent,
    StopReason,
    TextEvent,
    ThinkingChunkEvent,
    ThinkingEvent,
    UserPromptEvent,
)
from .messages import (
    AssistantMessage,
    CodeExecutionMessage,
    ExecutionResultMessage,
    Message,
    MessageRole,
    SummaryAcknowledgementMessage,
    SummaryMessage,
    SystemMessage,
    UserMessage,
)
from .models import (
    LiteLLMModel,
    Model,
    ModelError,
    ModelResponse,
    OpenAIModel,
    PromptTooLongError,
    ProviderBillingError,
    ProviderError,
    StreamStalledError,
    TokenUsage,
)
from .runtime import (
    BaseRuntime,
    Function,
    IPythonRuntime,
    PreemptibleRuntime,
    Runtime,
    RuntimeExecutionError,
    Type,
    Variable,
)
from .security import (
    AttributeRule,
    FunctionRule,
    ImportRule,
    RegexRule,
    SecurityChecker,
    SecurityError,
    SecurityRule,
    SecurityViolation,
)
from .skills import Skill, SkillDiscovery, SkillRegistry

if TYPE_CHECKING:
    from .runtime import IPyKernelRuntime


def __getattr__(name: str):
    if name == "IPyKernelRuntime":
        from .runtime import IPyKernelRuntime

        return IPyKernelRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Agent
    "CaveAgent",
    "AgentResponse",
    "Compactor",
    # Models
    "Model",
    "ModelResponse",
    "TokenUsage",
    "OpenAIModel",
    "LiteLLMModel",
    "ModelError",
    "PromptTooLongError",
    "ProviderBillingError",
    "ProviderError",
    "StreamStalledError",
    # Messages
    "Message",
    "MessageRole",
    "SummaryAcknowledgementMessage",
    "SummaryMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "CodeExecutionMessage",
    "ExecutionResultMessage",
    # Events
    "Event",
    "StopReason",
    "StatusType",
    "UserPromptEvent",
    "TextEvent",
    "ThinkingChunkEvent",
    "ThinkingEvent",
    "CodeEvent",
    "ExecutionResultEvent",
    "ExecutionTimeoutEvent",
    "SecurityErrorEvent",
    "StatusEvent",
    "FinalResponseEvent",
    "StoppedEvent",
    # Runtime
    "Runtime",
    "PreemptibleRuntime",
    "BaseRuntime",
    "RuntimeExecutionError",
    "IPythonRuntime",
    "IPyKernelRuntime",
    "Function",
    "Variable",
    "Type",
    # Security
    "SecurityChecker",
    "SecurityError",
    "SecurityViolation",
    "SecurityRule",
    "ImportRule",
    "FunctionRule",
    "AttributeRule",
    "RegexRule",
    # Skills
    "Skill",
    "SkillDiscovery",
    "SkillRegistry",
]
