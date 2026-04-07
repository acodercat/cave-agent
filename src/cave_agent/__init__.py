from .agent import CaveAgent, Message, MessageRole, EventType
from .display import with_display, render_events, render_user_prompt
from .models import Model, ModelResponse, TokenUsage, OpenAIServerModel, LiteLLMModel
from .runtime import Runtime, IPythonRuntime, Function, Variable, Type
from .security import SecurityChecker, SecurityError, SecurityViolation, SecurityRule, ImportRule, FunctionRule, AttributeRule, RegexRule
from .skills import Skill, SkillDiscovery, SkillRegistry


def __getattr__(name: str):
    if name == "IPyKernelRuntime":
        from .runtime import IPyKernelRuntime
        return IPyKernelRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CaveAgent",
    "Model",
    "ModelResponse",
    "TokenUsage",
    "OpenAIServerModel",
    "LiteLLMModel",
    "Message",
    "MessageRole",
    "EventType",
    "Runtime",
    "IPythonRuntime",
    "IPyKernelRuntime",
    "Function",
    "Variable",
    "Type",
    "SecurityChecker",
    "SecurityError",
    "SecurityViolation",
    "SecurityRule",
    "ImportRule",
    "FunctionRule",
    "AttributeRule",
    "RegexRule",
    "Skill",
    "SkillDiscovery",
    "SkillRegistry",
    "with_display",
    "render_events",
    "render_user_prompt",
]
