"""Regression tests for the audit fixes.

Covers:
- Runtime namespace isolation (independent InteractiveShell per runtime)
- Security hardening: submodule imports, call aliasing, whole-module regex
- Retry classification (authoritative HTTP status)
- Compaction token estimate + full_compact_needed prediction
- Function construction robustness for signature-less callables
"""

import pytest

from cave_agent.compaction import Compactor
from cave_agent.compaction.tokens import estimate_tokens
from cave_agent.messages import SystemMessage, UserMessage
from cave_agent.models.retry import is_retryable
from cave_agent.runtime import Function, IPythonRuntime, Variable
from cave_agent.security import (
    FunctionRule,
    ImportRule,
    RegexRule,
    SecurityChecker,
)
from cave_agent.skills import Skill


def _wire_skills(runtime, *skills):
    """Inject skills into a runtime the way ``CaveAgent._init_skills`` does."""
    from cave_agent.runtime.builtins import activate_skill
    from cave_agent.skills import SkillRegistry

    registry = SkillRegistry()
    registry.add_skills(list(skills))
    bindings = []
    for skill in skills:
        for function in skill.functions:
            bindings.append((function.name, function.func))
        for variable in skill.variables:
            bindings.append((variable.name, variable.value))
        for type_obj in skill.types:
            bindings.append((type_obj.name, type_obj.value))
    bindings.append(("_skill_store", registry.build_skill_store()))
    runtime.inject_resources(
        functions=[Function(activate_skill)],
        bindings=bindings,
    )


class TestRuntimeIsolation:
    @pytest.mark.asyncio
    async def test_separate_runtimes_have_isolated_namespaces(self):
        rt1 = IPythonRuntime(variables=[Variable("secret", "one")])
        rt2 = IPythonRuntime()

        # rt2 was never given `secret` — it must not see rt1's namespace.
        result = await rt2.execute("print(secret)")
        assert not result.success

        # rt1 still has its own variable.
        r1 = await rt1.execute("print(secret)")
        assert r1.success and "one" in (r1.stdout or "")

    @pytest.mark.asyncio
    async def test_same_named_variable_does_not_collide(self):
        rt1 = IPythonRuntime()
        rt2 = IPythonRuntime()
        await rt1.execute("x = 111")
        await rt2.execute("x = 222")
        a = await rt1.execute("print(x)")
        b = await rt2.execute("print(x)")
        assert (a.stdout or "").strip() == "111"
        assert (b.stdout or "").strip() == "222"

    def test_shell_objects_are_distinct(self):
        rt1 = IPythonRuntime()
        rt2 = IPythonRuntime()
        assert rt1._executor._shell is not rt2._executor._shell

    @pytest.mark.asyncio
    async def test_activate_skill_works_with_isolated_shell(self):
        # Regression: the non-singleton shell makes the global get_ipython()
        # return None, so activate_skill must resolve the namespace another way.
        def greet(name):
            return f"hi {name}"

        skill = Skill(
            name="greeter",
            description="greets",
            body_content="USE greet()",
            functions=[Function(greet)],
        )
        rt = IPythonRuntime()
        _wire_skills(rt, skill)

        r = await rt.execute('print(activate_skill("greeter"))')
        assert r.success and "USE greet()" in (r.stdout or "")

        # Exported function is injected into the namespace and persists.
        r2 = await rt.execute('print(greet("bob"))')
        assert r2.success and "hi bob" in (r2.stdout or "")

        # A separate runtime does not share the skill store.
        rt2 = IPythonRuntime()
        r3 = await rt2.execute('activate_skill("greeter")')
        assert not r3.success


class TestSecurityHardening:
    def _checker(self):
        return SecurityChecker(
            [
                ImportRule({"os", "subprocess"}),
                FunctionRule({"eval", "open"}),
            ]
        )

    def test_submodule_import_is_blocked(self):
        chk = self._checker()
        # `import os.path` binds top-level `os` — must be caught.
        assert chk.check_code("import os.path")
        assert chk.check_code("import os.path as p")
        assert chk.check_code("from os.path import join")

    def test_plain_forbidden_import_still_blocked(self):
        assert self._checker().check_code("import os")

    def test_unrelated_import_allowed(self):
        # "position" starts with "os"-ish text but is a different package.
        assert not self._checker().check_code("import math")
        assert not self._checker().check_code("import posixpath")

    def test_call_aliasing_is_blocked(self):
        chk = self._checker()
        assert chk.check_code("f = open\nf('/etc/passwd')")
        assert chk.check_code("sorted(x, key=eval)")

    def test_regex_scans_whole_module_not_just_expressions(self):
        chk = SecurityChecker([RegexRule(r"delete", "no delete")])
        assert chk.check_code("y = delete_everything()")  # assignment RHS
        assert chk.check_code("import delete")  # import statement
        assert not chk.check_code("x = keep()")

    def test_regex_reports_once(self):
        chk = SecurityChecker([RegexRule(r"print", "no print")])
        violations = chk.check_code("print(1)\nprint(2)\nprint(3)")
        assert len(violations) == 1


class TestRetryClassification:
    class _APIError(Exception):
        def __init__(self, message, status_code):
            super().__init__(message)
            self.status_code = status_code

    def test_retryable_status_retries(self):
        assert is_retryable(self._APIError("rate limited", 429))
        assert is_retryable(self._APIError("server error", 503))

    def test_nonretryable_status_is_authoritative(self):
        # A 400 whose text mentions "timeout" must NOT be retried.
        assert not is_retryable(self._APIError("invalid timeout parameter", 400))
        assert not is_retryable(self._APIError("connection field invalid", 401))

    def test_connectionless_errors_use_message_heuristic(self):
        assert is_retryable(ConnectionError("boom"))
        assert is_retryable(Exception("connection reset by peer"))
        assert not is_retryable(Exception("totally unrelated"))


def _compactor(context_window: int = 128_000) -> Compactor:
    """A compactor whose model declares nothing, so the fallback reserve applies."""
    from .fakes import FakeModel

    return Compactor(FakeModel(), context_window=context_window)


class TestCompactionEstimates:
    def test_estimate_anchors_to_the_api_count(self):
        from cave_agent.compaction.tokens import TokenAnchor

        msgs = [UserMessage("x" * 400)]  # heuristic ~100 tokens
        assert estimate_tokens(msgs, TokenAnchor(5000, estimate_tokens(msgs))) == 5000
        # An anchor taken against a larger history no longer describes this one.
        assert estimate_tokens(msgs, TokenAnchor(1, 400)) == 100

    def test_full_compact_not_needed_when_small(self):
        msgs = [SystemMessage("sys"), UserMessage("hi")]
        _, needs = _compactor().microcompact(msgs)
        assert needs is False

    def test_full_compact_needed_when_huge(self):
        # One enormous user message that microcompact cannot shrink.
        msgs = [SystemMessage("sys"), UserMessage("x" * 2_000_000)]
        _, needs = _compactor().microcompact(msgs)
        assert needs is True


class TestFunctionRobustness:
    def test_signatureless_callable_does_not_crash(self):
        # Some C builtins expose no introspectable signature.
        fn = Function(len)
        assert fn.name == "len"
        assert "(" in fn.signature


class TestSecurityChecksFailClosed:
    """A check that could not run has not found the code clean.

    `RegexRule` used to scan `ast.unparse` of the module, which recurses per
    node; a long chained expression blew the stack, the failure was swallowed,
    and every regex rule silently stopped applying to that cell.
    """

    def _checker(self):
        from cave_agent.security import FunctionRule, RegexRule, SecurityChecker

        return SecurityChecker([RegexRule(r"rm -rf"), FunctionRule({"system"})])

    def test_a_deep_expression_does_not_disable_the_regex_rule(self):
        deep = "x = " + "+".join(["1"] * 2000) + "\nimport os\nos.system('rm -rf /')"
        plain = "import os\nos.system('rm -rf /')"

        assert len(self._checker().check_code(deep)) == len(self._checker().check_code(plain))

    def test_clean_code_is_still_clean(self):
        assert self._checker().check_code("x = 1") == []

    def test_a_rule_that_raises_is_reported_not_skipped(self):
        from cave_agent.security import SecurityChecker
        from cave_agent.security.rules import SecurityRule

        class Broken(SecurityRule):
            def check(self, node):
                raise RuntimeError("rule exploded")

        violations = SecurityChecker([Broken()]).check_code("x = 1")

        assert violations, "a check that could not run must not report clean"
        assert "could not be evaluated" in violations[0].message
