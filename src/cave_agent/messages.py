"""Conversation message types.

The agent keeps history in these objects, not in wire dicts. Two of the five
roles are internal — ``CODE_EXECUTION`` and ``EXECUTION_RESULT`` describe the
code/result rhythm the agent loop is built on — and are mapped to plain
``assistant`` / ``user`` roles by :func:`to_wire` at the API boundary.

Compaction selects on these subclasses (microcompaction clears
:class:`ExecutionResultMessage` bodies specifically, and re-compaction finds
the previous :class:`SummaryMessage`), so any code that rebuilds history must
preserve the subclass rather than flattening everything to
:class:`UserMessage` / :class:`AssistantMessage`.
"""

from __future__ import annotations

from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CODE_EXECUTION = "code_execution"
    EXECUTION_RESULT = "execution_result"


# Internal roles -> the wire roles a provider understands.
_ROLE_MAP = {
    MessageRole.CODE_EXECUTION: MessageRole.ASSISTANT,
    MessageRole.EXECUTION_RESULT: MessageRole.USER,
}


class Message:
    """Base class for all message types in the agent conversation."""

    def __init__(self, content: str, role: MessageRole):
        self.content = content
        self.role = role

    @property
    def wire_role(self) -> str:
        """The role a provider API should see for this message."""
        return _ROLE_MAP.get(self.role, self.role).value

    def __repr__(self) -> str:
        preview = self.content[:40].replace("\n", " ")
        suffix = "…" if len(self.content) > 40 else ""
        return f"{type(self).__name__}({preview!r}{suffix})"


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


class SummaryMessage(UserMessage):
    """A compaction summary this agent generated.

    Presented to the model as a user turn, and identified by its **type**. The
    distinction is load-bearing: a summary is delivered into a conversation
    whose other participant also writes text, so any syntax marking it can be
    reproduced in an ordinary request. A type cannot be typed.

    ``body`` is the summary; ``content`` is the same text wrapped so the model
    can tell a summary from an instruction. Carried rather than re-parsed, so
    identification and extraction cannot disagree. Built in one place —
    ``Compactor._build_summary_messages``.
    """

    def __init__(self, content: str, body: str):
        super().__init__(content)
        self.body = body


class SummaryAcknowledgementMessage(AssistantMessage):
    """The synthetic assistant turn paired with a :class:`SummaryMessage`.

    Typed for the same reason the summary is: its text is a plausible thing for
    a model to actually say, so matching on the words alone both deletes genuine
    turns and lets a forged summary corroborate itself.
    """


class CodeExecutionMessage(Message):
    """An assistant turn that ended in a code block the agent executed."""

    def __init__(self, content: str):
        super().__init__(content, MessageRole.CODE_EXECUTION)


class ExecutionResultMessage(Message):
    """The runtime's response to a :class:`CodeExecutionMessage`."""

    def __init__(self, content: str):
        super().__init__(content, MessageRole.EXECUTION_RESULT)


def to_wire(messages: list[Message]) -> list[dict[str, str]]:
    """Convert internal messages to the provider dict format."""
    return [{"role": m.wire_role, "content": m.content} for m in messages]
