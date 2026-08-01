"""The runtime contract the agent depends on.

Pure protocol definitions with no execution-backend imports. The agent uses
this contract for caller-supplied runtimes; its convenience default still
constructs ``IPythonRuntime``. New backends plug in by satisfying the protocol
structurally; no subclassing required. :class:`BaseRuntime` exists as a
convenience for backends that want the shared registry/description logic, not
as a requirement.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .executor import ExecutionResult
from .primitives import Function, Type, Variable


@runtime_checkable
class Runtime(Protocol):
    """A stateful Python runtime that executes LLM-generated code.

    The defining property is persistence: variables, imports and function
    definitions survive across :meth:`execute` calls, so injected objects stay
    live between turns instead of being serialized in and out of the context
    window.
    """

    async def execute(self, code: str) -> ExecutionResult:
        """Run *code* in the persistent namespace and capture its output.

        If cancellation reaches the backend, it must finish request-scoped
        cleanup before re-raising. Only the backend can target the request it
        admitted; a caller sending an interrupt after ``execute`` unwinds may
        hit unrelated work that acquired the runtime next.
        """
        ...

    async def retrieve(self, name: str) -> Any:
        """Read back a registered variable's current value.

        Raises ``KeyError`` for a name this runtime does not manage — use
        :meth:`get_from_namespace` for arbitrary namespace reads.
        """
        ...

    async def reset(self) -> None:
        """Clear the namespace and re-inject registered resources.

        Atomic from a caller's point of view: no execution observes the
        namespace between the clear and the restore.
        """
        ...

    async def bind_unique(self, prefix: str, value: Any, *, start: int = 1) -> str:
        """Bind *value* under an unused ``prefix_N`` name and return the name.

        Part of the contract because only the runtime can see the namespace the
        generated code runs in — a caller allocating names cannot tell whether
        the model already made one — and because two agents may share a
        runtime. Implementations must make the check and the bind one
        indivisible step.
        """
        ...

    def inject_function(self, function: Function) -> None:
        """Register a function and make it callable from generated code.

        Part of the contract because the agent registers into the runtime
        itself: enabling skills injects the built-in ``activate_skill``.
        """
        ...

    def inject_variable(self, variable: Variable) -> None:
        """Register a variable and bind its value."""
        ...

    def inject_type(self, type_obj: Type) -> None:
        """Register a type and bind the class."""
        ...

    def inject_resources(
        self,
        *,
        functions: list[Function] | None = None,
        variables: list[Variable] | None = None,
        types: list[Type] | None = None,
        bindings: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Register a group only after every name has been validated.

        A failure must leave both the registry and namespace unchanged.
        """
        ...

    def inject_into_namespace(
        self,
        name: str,
        value: Any,
        *,
        replace: bool = True,
    ) -> None:
        """Bind *value* without describing it to the LLM.

        ``replace=False`` reserves a new binding and raises on any collision.
        """
        ...

    async def get_from_namespace(self, name: str) -> Any:
        """Read any name from the namespace, registered or not.

        Raises ``KeyError`` when the name is not bound. ``None`` is a value a
        variable can genuinely hold, so it cannot also mean "absent" — a
        backend that used it for both left callers unable to tell the two
        apart, and made the two backends answer the same question differently.
        """
        ...

    def describe_functions(self) -> str: ...

    def describe_variables(self) -> str: ...

    def describe_types(self) -> str: ...


@runtime_checkable
class PreemptibleRuntime(Runtime, Protocol):
    """A runtime whose in-flight execution can actually be stopped.

    Defining :meth:`interrupt` *is* the capability claim — it promises that a
    caller who hits a deadline can stop the code that is running. A backend
    that runs generated code in this process cannot promise that and must not
    define it; :class:`~cave_agent.agent.CaveAgent` refuses ``max_exec_timeout``
    without this protocol.

    Expressing the capability as a method rather than a ``preemptible: bool``
    is deliberate: a Protocol data member is a *required* member, so every
    structural implementer would have to add a line just to say "no", and the
    method already carries the meaning. Implementing ``interrupt`` is the
    claim — a no-op body would be a promise a caller cannot rely on, and
    ``max_exec_timeout`` relies on it.
    """

    async def interrupt(self) -> None:
        """Stop in-flight execution."""
        ...
