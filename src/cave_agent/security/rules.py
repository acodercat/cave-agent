import ast
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Set, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass()
class SecurityViolation:
    """Represents a security violation found in code."""
    message: str


class SecurityRule(ABC):
    """Abstract base class for security rules.

    All security rules must inherit from this class and implement
    the check method to analyze AST nodes for violations.
    """

    @abstractmethod
    def check(self, node: ast.AST) -> List[SecurityViolation]:
        """Check if the AST node violates this rule.

        Args:
            node: AST node to analyze

        Returns:
            List of violations found (empty if none)
        """
        pass


class ImportRule(SecurityRule):
    """Rule to detect forbidden imports.

    Matches on the top-level package, so forbidding ``"os"`` also blocks
    ``import os.path`` (which binds ``os`` in the namespace) and
    ``from os.path import join``.
    """

    def __init__(self, forbidden_modules: Set[str]):
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

    def check(self, node: ast.AST) -> List[SecurityViolation]:
        violations = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                if self._is_forbidden(alias.name):
                    violations.append(SecurityViolation(
                        message=f"Forbidden import detected: {alias.name} at line {node.lineno}",
                    ))

        elif isinstance(node, ast.ImportFrom):
            # node.module is None for relative imports like "from . import x"
            if self._is_forbidden(node.module):
                violations.append(SecurityViolation(
                    message=f"Forbidden import detected: from {node.module} at line {node.lineno}",
                ))

        return violations


class FunctionRule(SecurityRule):
    """Rule to detect forbidden function calls."""

    def __init__(self, forbidden_functions: Set[str], description: Optional[str] = None):
        self.description = description
        self.forbidden_functions = forbidden_functions

    def check(self, node: ast.AST) -> List[SecurityViolation]:
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
                    violations.append(self._violation(
                        f"Forbidden function call '{func_name}' at line {child.lineno}"))
            # Aliasing / indirection, e.g. `f = open` or `sorted(key=eval)`: the
            # forbidden name is loaded but not the callee of a call handled above.
            elif (
                isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in self.forbidden_functions
                and child not in called_names
            ):
                violations.append(self._violation(
                    f"Forbidden reference to '{child.id}' at line {child.lineno}"))

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
        elif isinstance(func_node, ast.Attribute):
            # For calls like obj.method(), return the method name
            return func_node.attr
        elif isinstance(func_node, ast.Call):
            # For nested calls, recurse to find the innermost function
            return self._get_function_name(func_node.func)
        return ""


class AttributeRule(SecurityRule):
    """Rule to detect forbidden attribute access."""

    def __init__(self, forbidden_attributes: Set[str]):
        self.forbidden_attributes = forbidden_attributes

    def check(self, node: ast.AST) -> List[SecurityViolation]:
        violations = []

        if isinstance(node, ast.Attribute):
            if node.attr in self.forbidden_attributes:
                violations.append(SecurityViolation(
                    message=f"Forbidden attribute access detected: {node.attr} at line {node.lineno}",
                ))

        return violations


class RegexRule(SecurityRule):
    """Security rule using regex patterns.

    Matches the pattern against the unparsed source of the whole module,
    so it scans every statement — assignments, calls, imports, comprehensions —
    not just top-level expressions. Fires on the ``ast.Module`` node, so each
    match is reported once.
    """

    def __init__(self, pattern: str, description: Optional[str] = None):
        self.description = description if description else f"Regex rule: {pattern}"
        try:
            self.pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

    def check(self, node: ast.AST) -> List[SecurityViolation]:
        violations = []

        # ast.walk yields the Module node exactly once; unparse the whole
        # tree there so we scan all statements a single time.
        if isinstance(node, ast.Module):
            try:
                source = ast.unparse(node)
            except Exception:
                logger.debug("Failed to unparse module for regex check", exc_info=True)
                return violations

            if self.pattern.search(source):
                violations.append(SecurityViolation(
                    message=f"Security rule: {self.description}"
                ))

        return violations
