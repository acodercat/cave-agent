"""Message and event types used across the cave-agent package."""

from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CODE_EXECUTION = "code_execution"
    EXECUTION_RESULT = "execution_result"


# Maps internal roles to LLM-facing roles
_ROLE_MAP = {
    MessageRole.CODE_EXECUTION: MessageRole.ASSISTANT,
    MessageRole.EXECUTION_RESULT: MessageRole.USER,
}


class Message:
    """Base class for all message types in the agent conversation."""

    def __init__(self, content: str, role: MessageRole):
        self.content = content
        self.role = role


class SystemMessage(Message):
    """System message that provides instructions to the LLM."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.SYSTEM)


class UserMessage(Message):
    """Message from the user to the agent."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.USER)


class AssistantMessage(Message):
    """Message from the assistant (LLM) to the user."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.ASSISTANT)


class CodeExecutionMessage(Message):
    """Message representing code to be executed by the agent."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.CODE_EXECUTION)


class ExecutionResultMessage(Message):
    """Message representing the result from code execution."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.EXECUTION_RESULT)


class EventType(Enum):
    TEXT = "text"
    CODE = "code"
    EXECUTION_OUTPUT = "execution_output"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_OUTPUT_EXCEEDED = "execution_output_exceeded"
    FINAL_RESPONSE = "final_response"
    MAX_STEPS_REACHED = "max_steps_reached"
    SECURITY_ERROR = "security_error"
    COMPACTING = "compacting"
    COMPACTED = "compacted"


class Event:
    def __init__(self, event_type: EventType, content: str):
        self.type = event_type
        self.content = content
