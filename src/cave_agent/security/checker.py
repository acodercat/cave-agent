import ast
import logging

from .rules import SecurityRule, SecurityViolation

logger = logging.getLogger(__name__)


class SecurityChecker:
    """AST-based static screen for LLM-generated code.

    Applies rules to the parsed AST to catch obviously dangerous patterns
    (forbidden imports, calls, attribute access, regex matches) before code
    runs. This is **advisory hardening, not a sandbox**: static analysis
    cannot catch every obfuscation or indirection (dynamic attribute access,
    reflection, C-extension escapes, etc.). For untrusted code, run inside a
    real isolation boundary — a container with seccomp/gVisor and OS resource
    limits, or at minimum the process-isolated ``IPyKernelRuntime`` — and
    treat this checker as defense-in-depth on top of that.

    Example:
        >>> from cave_agent.security import SecurityChecker, ImportRule, FunctionRule, AttributeRule, RegexRule
        >>> checker = SecurityChecker([
        >>>     ImportRule(set(["os", "subprocess", "sys", "shutil", "pathlib", "socket", "urllib", "http", "ctypes", "gc", "csv"])),
        >>>     FunctionRule(set(["eval", "exec", "compile", "open", "input", "raw_input", "exit", "quit", "__import__", "globals", "locals", "breakpoint"])),
        >>>     AttributeRule(set(["__globals__", "__locals__", "__code__", "__closure__", "__defaults__", "__dict__", "__class__", "__bases__", "__mro__", "__subclasses__", "__import__", "__builtins__"])),
        >>>     RegexRule(r"delete", "Detects forbidden statements")
        >>> ])
        >>> violations = checker.check_code("import os; os.system('ls')")
        >>> print(len(violations))
    """

    def __init__(self, rules: list[SecurityRule]):
        """Initialize SecurityChecker with specified rules.

        Args:
            rules: List of security rules

        """
        self.rules = []
        for rule in rules:
            self.add_rule(rule)

    def add_rule(self, rule: SecurityRule):
        """Add a security rule.

        Args:
            rule: Security rule to add

        """

        self.rules.append(rule)

    def check_code(self, code: str) -> list[SecurityViolation]:
        """Analyze Python code for security violations.

        Parses the code into an AST and applies all security rules
        to detect security issues.

        Args:
            code: Python code string to analyze

        Returns:
            List of SecurityViolation containing all violations found

        """
        violations = []
        if not code or not code.strip():
            violations.append(
                SecurityViolation(
                    message="Parse error: Code cannot be empty",
                )
            )
            return violations

        try:
            # Parse code into AST
            tree = ast.parse(code)
        except SyntaxError as error:
            violations.append(
                SecurityViolation(
                    message=f"Syntax error: {error}",
                )
            )
            return violations
        except Exception as error:
            violations.append(
                SecurityViolation(
                    message=f"Parse error: {error}",
                )
            )
            return violations

        # Text rules see the source once, before the walk.
        for rule in self.rules:
            violations.extend(self._apply(rule, rule.check_source, code))

        # Analyze AST with all rules
        for node in ast.walk(tree):
            for rule in self.rules:
                violations.extend(self._apply(rule, rule.check, node))

        return violations

    @staticmethod
    def _apply(rule, check, argument) -> list[SecurityViolation]:
        """Run one check, reporting a failure *as a violation*.

        A check that could not run has not found the code clean — it has found
        out nothing. Skipping it silently disabled the rule for that cell while
        the caller saw a clean report, which is the one outcome a security gate
        must never produce.
        """
        try:
            return check(argument)
        except Exception as error:
            logger.warning("Rule %r failed", rule, exc_info=True)
            return [
                SecurityViolation(
                    message=f"Security rule could not be evaluated: {rule!r} ({error})",
                )
            ]


class SecurityError(Exception):
    """Exception raised when code fails security checks."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
