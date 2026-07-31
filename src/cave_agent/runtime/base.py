"""BaseRuntime — shared registry and description logic for runtime backends.

Holds the LLM-facing ``functions`` / ``variables`` / ``types`` registries and
renders them into the strings the system prompt slots in. Execution itself is
delegated to an *executor* passed to the constructor.

The executor is a constructor argument rather than an attribute subclasses are
expected to set beforehand: the previous arrangement required every subclass to
assign ``self._executor`` *before* calling ``super().__init__()``, an implicit
ordering contract that fails with an ``AttributeError`` several frames away
from the mistake.
"""

from __future__ import annotations

import copy
import inspect
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from types import NoneType
from typing import Any, ForwardRef, get_args, get_origin, get_type_hints

from .._placeholders import normalize_identifier, usable_identifier
from .executor import ExecutionResult, RuntimeExecutionError
from .primitives import Function, Type, Variable


class BaseRuntime:
    """Convenience base for runtimes that share the standard registry.

    Satisfies the :class:`~cave_agent.runtime.protocol.Runtime` protocol.
    """

    # Built-in types that should not be auto-injected.
    _BUILTIN_TYPES = frozenset(
        {
            str,
            int,
            float,
            bool,
            bytes,
            bytearray,
            list,
            dict,
            tuple,
            set,
            frozenset,
            NoneType,
            object,
            type,
        }
    )

    def __init__(
        self,
        executor: Any,
        functions: list[Function] | None = None,
        variables: list[Variable] | None = None,
        types: list[Type] | None = None,
    ):
        """
        Args:
            executor: Backend that actually runs code. Must expose
                ``execute``, ``inject_many_into_namespace``,
                ``get_from_namespace`` and ``reset``.
            functions: Functions to inject and describe to the LLM.
            variables: Variables to inject and describe to the LLM.
            types: Types to inject; only these get schemas in the prompt.
        """
        self._executor = executor
        self._functions: dict[str, Function] = {}
        self._variables: dict[str, Variable] = {}
        self._types: dict[str, Type] = {}
        # Raw, non-LLM-facing bindings, kept so reset() can restore them.
        self._namespace: dict[str, Any] = {}
        # Names selected by an asynchronous bind but not committed yet. Keeping
        # reservations separate from the namespace lets synchronous registration
        # reject the same name without advertising a value that has not landed.
        self._reservations: set[str] = set()
        # Registration is synchronous and may be called from any thread.  The
        # snapshot, validation, backend staging and registry commit form one
        # transaction; otherwise two callers can both claim the same name.
        self._registry_lock = threading.RLock()

        self.inject_resources(
            types=types,
            functions=functions,
            variables=variables,
        )

    def _claim_name(
        self,
        name: str,
        kind: str,
        *,
        taken: set[str] | None = None,
    ) -> str:
        """Validate and normalize *name*, and reserve it across all registries.

        Every registered resource ends up in one Python namespace, so the
        registries cannot each police their own: two entries that normalize to
        the same identifier bind the same variable, and the runtime advertises
        both while only one exists. Every route that binds a name goes through
        here — including auto-injected types and raw bindings, which used to
        overwrite explicit registrations without a word.
        """
        normalized = usable_identifier(name, kind)
        claimed = self._bound_names() if taken is None else taken
        if normalized in claimed:
            if normalized == name:
                raise ValueError(f"{kind} '{name}' already exists")
            raise ValueError(
                f"{kind} '{name}' collides with the existing name '{normalized}' "
                "after NFKC normalization"
            )
        if taken is not None:
            taken.add(normalized)
        return normalized

    def _bound_names(self) -> set[str]:
        """Every name this runtime has claimed, in the one namespace they share."""
        with self._registry_lock:
            return {
                *self._functions,
                *self._variables,
                *self._types,
                *self._namespace,
                *self._reservations,
            }

    def inject_function(self, function: Function):
        """Register and bind a function.

        Also auto-injects custom types found in the signature, with schemas
        hidden — they become usable in generated code without bloating the
        prompt. Use an explicit :class:`Type` to show a schema.
        """
        self.inject_resources(functions=[function])

    def inject_variable(self, variable: Variable):
        """Register and bind a variable, auto-injecting its type (schema hidden)."""
        self.inject_resources(variables=[variable])

    def inject_type(self, type_obj: Type):
        """Register and bind a type/class."""
        self.inject_resources(types=[type_obj])

    def inject_resources(
        self,
        *,
        functions: list[Function] | None = None,
        variables: list[Variable] | None = None,
        types: list[Type] | None = None,
        bindings: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Validate and register a group of resources as one operation.

        All names are checked before any registry or namespace is changed. The
        runtime also keeps its own shallow copy of each descriptor: updating a
        variable in one runtime must not mutate a descriptor another runtime
        was constructed from.
        """
        with self._registry_lock:
            claimed = self._bound_names()

            prepared_types: list[Type] = []
            for original in types or []:
                type_obj = copy.copy(original)
                type_obj.name = self._claim_name(
                    type_obj.name,
                    "Type",
                    taken=claimed,
                )
                prepared_types.append(type_obj)

            prepared_functions: list[Function] = []
            for original in functions or []:
                function = copy.copy(original)
                function.name = self._claim_name(
                    function.name,
                    "Function",
                    taken=claimed,
                )
                prepared_functions.append(function)

            prepared_variables: list[Variable] = []
            for original in variables or []:
                variable = copy.copy(original)
                variable.name = self._claim_name(
                    variable.name,
                    "Variable",
                    taken=claimed,
                )
                prepared_variables.append(variable)

            prepared_bindings: list[tuple[str, Any]] = []
            for name, value in bindings or []:
                normalized = self._claim_name(name, "Binding", taken=claimed)
                prepared_bindings.append((normalized, value))

            staged = {
                **{type_obj.name: type_obj.value for type_obj in prepared_types},
                **{function.name: function.func for function in prepared_functions},
                **{variable.name: variable.value for variable in prepared_variables},
                **dict(prepared_bindings),
            }

            def commit() -> None:
                for type_obj in prepared_types:
                    self._types[type_obj.name] = type_obj
                for function in prepared_functions:
                    self._functions[function.name] = function
                for variable in prepared_variables:
                    self._variables[variable.name] = variable
                self._namespace.update(prepared_bindings)

            if staged:
                self._executor.inject_many_into_namespace(staged, commit=commit)
            else:
                commit()

            # Explicit resources now own their names, so inferred types can never
            # displace them.
            for function in prepared_functions:
                self._auto_inject_types_from_signature(function.func)
            for variable in prepared_variables:
                if variable.value is not None:
                    self._try_auto_inject_type(
                        type(variable.value),
                        include_schema=False,
                        include_doc=False,
                    )

    def update_variable(self, name: str, value: Any):
        """Replace a registered variable's value, found by its normalized name.

        Raises:
            KeyError: if the variable was never registered.
            TypeError: if the new value isn't an instance of the original's
                type (subclasses are allowed).
        """
        with self._registry_lock:
            name = normalize_identifier(name)
            if name not in self._variables:
                raise KeyError(
                    f"Variable '{name}' does not exist. "
                    f"Available variables: {list(self._variables.keys())}"
                )

            variable = self._variables[name]
            expected_type = variable.declared_type
            if (
                value is not None
                and expected_type is not None
                and not isinstance(value, expected_type)
            ):
                raise TypeError(
                    f"Cannot update variable '{name}': type mismatch. "
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )

            def commit() -> None:
                variable.value = value
                if value is not None and expected_type is None:
                    variable.declared_type = type(value)
                    variable.type_name = type(value).__name__

            self._executor.inject_many_into_namespace({name: value}, commit=commit)
            if value is not None and expected_type is None:
                self._try_auto_inject_type(
                    type(value),
                    include_schema=False,
                    include_doc=False,
                )

    def _try_auto_inject_type(
        self,
        cls: type,
        include_schema: bool = False,
        include_doc: bool = False,
    ) -> bool:
        """Inject *cls* if it is injectable and its name is free.

        Automatic injection never displaces an explicit registration: a caller
        who registered ``Variable("Widget", widget)`` and then a function taking
        a ``Widget`` had the instance silently replaced by the class. Nor does
        it raise — the caller did not ask for this type, so a taken name simply
        means there is nothing to add.
        """
        if not self._is_injectable_type(cls):
            return False
        try:
            name = usable_identifier(cls.__name__, "Type")
        except ValueError:
            # A class whose name cannot be written in Python — generated code
            # could never refer to it, and on the kernel it breaks the whole
            # injection batch. Declining is right: the caller asked to register
            # a *function*, not this type.
            return False
        if name in self._bound_names():
            return False
        candidate = Type(
            cls,
            include_schema=include_schema,
            include_doc=include_doc,
        )
        candidate.name = name

        def commit() -> None:
            self._types[name] = candidate

        try:
            self._executor.inject_many_into_namespace(
                {name: cls},
                commit=commit,
            )
        except Exception:
            # Inference is optional: an unsupported annotation must not prevent
            # the explicitly requested function or variable from registering.
            return False
        return True

    def _is_injectable_type(self, cls: type) -> bool:
        if not isinstance(cls, type):
            return False
        if cls in self._BUILTIN_TYPES:
            return False
        # Skip types without proper names (lambdas, locals, etc.)
        if not hasattr(cls, "__name__") or cls.__name__.startswith("<"):
            return False
        return True

    def _auto_inject_types_from_signature(self, func: Callable):
        try:
            resolved = get_type_hints(func, include_extras=True)
        except Exception:
            resolved = {}
        else:
            for type_hint in resolved.values():
                self._process_type_for_injection(type_hint)
            return

        try:
            annotations = inspect.get_annotations(func, eval_str=False)
        except (ValueError, TypeError):
            try:
                sig = inspect.signature(func)
            except (ValueError, TypeError):
                return
            annotations = {
                param.name: param.annotation
                for param in sig.parameters.values()
                if param.annotation != inspect.Parameter.empty
            }
            if sig.return_annotation != inspect.Signature.empty:
                annotations["return"] = sig.return_annotation

        globalns = getattr(func, "__globals__", None)
        if globalns is None:
            module = inspect.getmodule(func)
            globalns = vars(module) if module is not None else {}

        # Resolve each annotation independently. ``get_type_hints(func)`` is
        # all-or-nothing: one missing forward reference must not hide a valid
        # custom type elsewhere in the same signature.
        def probe() -> None:
            pass

        for annotation in annotations.values():
            probe.__annotations__ = {"value": annotation}
            try:
                resolved_annotation = get_type_hints(
                    probe,
                    globalns=globalns,
                    localns=globalns,
                    include_extras=True,
                )["value"]
            except Exception:
                continue
            self._process_type_for_injection(resolved_annotation)

    def _process_type_for_injection(self, type_hint: Any):
        """Walk a type hint, injecting any custom classes found."""
        if type_hint is None or type_hint is NoneType:
            return
        # String / ForwardRef annotations aren't resolvable here without the
        # defining module's namespace; skip rather than guess.
        if isinstance(type_hint, (str, ForwardRef)):
            return
        origin = get_origin(type_hint)
        if origin is not None:
            for arg in get_args(type_hint):
                if arg is not NoneType:
                    self._process_type_for_injection(arg)
            return
        if isinstance(type_hint, type):
            self._try_auto_inject_type(type_hint, include_schema=False, include_doc=False)

    def inject_into_namespace(
        self,
        name: str,
        value: Any,
        *,
        replace: bool = True,
    ) -> None:
        """Bind *value* to *name* without registering it as LLM-facing.

        Recorded so :meth:`reset` can replay it. Skills rely on this for both
        their instruction store and hidden exports; neither belongs in the
        LLM-facing registries, but both must survive a reset alongside the
        regular ``activate_skill`` function.

        Claims its name like every other binding: a raw injection that landed on
        a registered variable's name replaced it silently, and the prompt went
        on describing something that was no longer there. Re-binding a name this
        method itself owns is allowed by default; pass ``replace=False`` when
        registering a distinct resource which must not displace an existing raw
        binding.
        """
        with self._registry_lock:
            normalized = normalize_identifier(name)
            if normalized in self._namespace:
                if not replace:
                    raise ValueError(f"Binding '{normalized}' already exists")

                def commit() -> None:
                    self._namespace[normalized] = value

                self._executor.inject_many_into_namespace(
                    {normalized: value},
                    commit=commit,
                )
                return
            self.inject_resources(bindings=[(name, value)])

    async def get_from_namespace(self, name: str) -> Any:
        return await self._executor.get_from_namespace(name)

    async def bind_unique(self, prefix: str, value: Any, *, start: int = 1) -> str:
        """Bind *value* to an unused ``prefix_N`` name and return that name.

        Delegated to the executor: only the namespace the generated code runs
        in knows which names are free, and a registry here would miss every
        variable the model made for itself. Each backend performs check-and-bind
        as one indivisible step, so agents sharing a runtime cannot collide.

        *start* is a floor the caller can raise. Names promised to a model in an
        earlier run live in the conversation, not in this namespace, and
        reusing one would re-point a marker the model still trusts.

        The *synthesized* name is validated here, at the boundary every backend
        shares — not the prefix, which is not what gets bound. A prefix that
        cannot form one gave the model a pointer it could not follow in-process
        and a syntax error on the kernel.
        """
        if not isinstance(start, int) or start < 1:
            raise ValueError(f"start must be a positive integer, not {start!r}")
        prefix = usable_identifier(f"{prefix}_{start}", "Prefix").rsplit("_", 1)[0]

        def reserve(name: str) -> bool:
            with self._registry_lock:
                if name in self._bound_names():
                    return False
                self._reservations.add(name)
                return True

        def record(name: str) -> None:
            with self._registry_lock:
                self._reservations.remove(name)
                self._namespace[name] = value

        def release(name: str) -> None:
            with self._registry_lock:
                self._reservations.discard(name)

        async with self._containing_backend_failures():
            return await self._executor.bind_unique(
                prefix,
                value,
                start=start,
                reserve=reserve,
                on_bound=record,
                on_failed=release,
            )

    @asynccontextmanager
    async def _containing_backend_failures(self):
        """Turn anything an executor raises into a :class:`RuntimeExecutionError`.

        Wraps the operations the *agent loop* drives — ``execute`` and
        ``bind_unique`` — so a new backend cannot leak an untyped exception
        into it by forgetting to wrap; an untyped escape there reads as a
        defect in the agent rather than a failing runtime.

        ``retrieve`` and ``get_from_namespace`` are deliberately outside it:
        they are caller-facing reads whose ``KeyError`` is their contract, and
        the loop never calls them.
        """
        try:
            yield
        except RuntimeExecutionError:
            raise
        except Exception as error:
            raise RuntimeExecutionError(f"{type(error).__name__}: {error}") from error

    async def execute(self, code: str) -> ExecutionResult:
        """Run *code*, containing backend failures."""
        async with self._containing_backend_failures():
            return await self._executor.execute(code)

    async def retrieve(self, name: str) -> Any:
        # Normalized, not claimed: this is a lookup, and it has to agree with
        # the name registration stored — which is the one Python binds.
        name = normalize_identifier(name)
        with self._registry_lock:
            if name not in self._variables:
                raise KeyError(
                    f"Variable '{name}' is not managed by this runtime. "
                    f"Available variables: {list(self._variables.keys())}"
                )
        return await self._executor.get_from_namespace(name)

    async def reset(self) -> None:
        """Clear the namespace and restore every registered resource, atomically.

        The registry survives — a reset runtime is still the runtime the system
        prompt describes. Dropping the registrations instead would leave the
        agent advertising functions that no longer exist.

        The restorations are handed to the executor rather than replayed after
        it returns, so no execution can be admitted into the gap where the
        namespace is genuinely empty.
        """
        await self._executor.reset(self._registered_bindings)

    def _registered_bindings(self) -> dict[str, Any]:
        """Everything a freshly reset namespace must contain.

        Raw bindings are included: a registered function whose supporting data
        went missing is worse than either being absent.
        """
        with self._registry_lock:
            return {
                **{t.name: t.value for t in self._types.values()},
                **{f.name: f.func for f in self._functions.values()},
                **{v.name: v.value for v in self._variables.values()},
                **self._namespace,
            }

    def describe_variables(self) -> str:
        with self._registry_lock:
            if not self._variables:
                return "No variables available"
            return "\n".join(str(v) for v in self._variables.values())

    def describe_functions(self) -> str:
        with self._registry_lock:
            if not self._functions:
                return "No functions available"
            return "\n".join(str(f) for f in self._functions.values())

    def describe_types(self) -> str:
        """Render schemas for the types that asked to be shown.

        Auto-injected types (from signatures and variable values) render to the
        empty string and are skipped — they are usable but invisible.
        """
        with self._registry_lock:
            if not self._types:
                return "No types available"
            schemas = [s for s in (str(t) for t in self._types.values()) if s]
            return "\n".join(schemas) if schemas else "No types available"
