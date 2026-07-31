import ast
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityViolation:
    """Represents a security violation found in code."""

    message: str


class SecurityRule(ABC):
    """Abstract base class for security rules.

    All security rules must inherit from this class and implement
    the check method to analyze AST nodes for violations.
    """

    @abstractmethod
    def check(self, node: ast.AST) -> list[SecurityViolation]:
        """Check if the AST node violates this rule.

        Args:
            node: AST node to analyze

        Returns:
            List of violations found (empty if none)
        """
        ...

    def check_source(self, source: str) -> list[SecurityViolation]:
        """Check the raw source once, before the AST walk.

        For rules that are about *text* rather than structure. The default does
        nothing, so structural rules need not care.
        """
        return []


class ImportRule(SecurityRule):
    """Rule to detect forbidden imports.

    Matches on the top-level package, so forbidding ``"os"`` also blocks
    ``import os.path`` (which binds ``os`` in the namespace) and
    ``from os.path import join``.
    """

    def __init__(self, forbidden_modules: set[str]):
        self.forbidden_modules = forbidden_modules

    def _is_forbidden(self, module: str | None) -> bool:
        """True if *module* or any of its parent packages is forbidden."""
        if not module:
            return False
        # "os.path" -> check "os.path", "os"
        parts = module.split(".")
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in self.forbidden_modules:
                return True
        return False

    def check(self, node: ast.AST) -> list[SecurityViolation]:
        violations = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                if self._is_forbidden(alias.name):
                    violations.append(
                        SecurityViolation(
                            message=f"Forbidden import detected: {alias.name} at line {node.lineno}",
                        )
                    )

        elif isinstance(node, ast.ImportFrom):
            # node.module is None for relative imports like "from . import x"
            if self._is_forbidden(node.module):
                violations.append(
                    SecurityViolation(
                        message=f"Forbidden import detected: from {node.module} at line {node.lineno}",
                    )
                )

        return violations


class FunctionRule(SecurityRule):
    """Rule to detect forbidden function calls."""

    def __init__(self, forbidden_functions: set[str], description: str | None = None):
        self.description = description
        self.forbidden_functions = forbidden_functions

    def check(self, node: ast.AST) -> list[SecurityViolation]:
        # Fire once on the module root and walk it here, so each forbidden name
        # is judged in context. The checker's own ``ast.walk`` visits a call and
        # its callee ``Name`` as separate nodes, so a per-node rule would report
        # ``open(...)`` twice — once as a call, once as a reference. Handling the
        # whole tree in one pass lets a direct call and a bare alias
        # (``f = open``, ``sorted(key=eval)``) each be reported exactly once.
        if not isinstance(node, ast.Module):
            return []

        violations = []
        called_names = {
            child.func
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = self._get_function_name(child.func)
                if func_name in self.forbidden_functions:
                    violations.append(
                        self._violation(
                            f"Forbidden function call '{func_name}' at line {child.lineno}"
                        )
                    )
            # Aliasing / indirection, e.g. `f = open` or `sorted(key=eval)`: the
            # forbidden name is loaded but not the callee of a call handled above.
            elif (
                isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in self.forbidden_functions
                and child not in called_names
            ):
                violations.append(
                    self._violation(f"Forbidden reference to '{child.id}' at line {child.lineno}")
                )

        return violations

    def _violation(self, message: str) -> SecurityViolation:
        if self.description:
            message += f": {self.description}"
        return SecurityViolation(message=message)

    def _get_function_name(self, func_node: ast.AST) -> str:
        """Extract function name from various call patterns.

        Handles Name, Attribute, and nested Call nodes to extract
        the actual function name being called.
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            # For calls like obj.method(), return the method name
            return func_node.attr
        if isinstance(func_node, ast.Call):
            # For nested calls, recurse to find the innermost function
            return self._get_function_name(func_node.func)
        return ""


class AttributeRule(SecurityRule):
    """Rule to detect forbidden attribute access."""

    def __init__(self, forbidden_attributes: set[str]):
        self.forbidden_attributes = forbidden_attributes

    def check(self, node: ast.AST) -> list[SecurityViolation]:
        violations = []

        if isinstance(node, ast.Attribute):
            if node.attr in self.forbidden_attributes:
                violations.append(
                    SecurityViolation(
                        message=f"Forbidden attribute access detected: {node.attr} at line {node.lineno}",
                    )
                )

        return violations


class RegexRule(SecurityRule):
    """Security rule using regex patterns.

    Matches against the code as written, once per check. It used to match
    against ``ast.unparse`` of the module instead, which recurses per node and
    blew the stack on a long chained expression — and the failure was swallowed,
    so every regex rule silently stopped applying to that cell. Reading the
    source needs no recursion and sees what the author actually wrote.
    """

    def __init__(self, pattern: str, description: str | None = None):
        self.description = description if description else f"Regex rule: {pattern}"
        try:
            self.pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
        except re.error as error:
            raise ValueError(f"Invalid regex pattern '{pattern}': {error}") from error

    def check(self, node: ast.AST) -> list[SecurityViolation]:
        """Structural checking is not this rule's business — see check_source."""
        return []

    def check_source(self, source: str) -> list[SecurityViolation]:
        if self.pattern.search(source):
            return [SecurityViolation(message=f"Security rule: {self.description}")]
        return []
