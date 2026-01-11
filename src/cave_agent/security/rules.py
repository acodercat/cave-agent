import ast
import re
from abc import ABC, abstractmethod
from typing import List, Set, Optional
from dataclasses import dataclass


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
    """Rule to detect forbidden imports."""

    def __init__(self, forbidden_modules: Set[str]):
        self.forbidden_modules = forbidden_modules

    def check(self, node: ast.AST) -> List[SecurityViolation]:
        violations = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in self.forbidden_modules:
                    violations.append(SecurityViolation(
                        message=f"Forbidden import detected: {alias.name} at line {node.lineno}",
                    ))

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module in self.forbidden_modules:
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
        violations = []

        if isinstance(node, ast.Call):
            func_name = self._get_function_name(node.func)
            if func_name in self.forbidden_functions:
                message = f"Forbidden function call '{func_name}' at line {node.lineno}"
                if self.description:
                    message += f": {self.description}"

                violations.append(SecurityViolation(message=message))

        return violations

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

    Allows users to define security rules by providing
    a regex pattern that matches against the string representation
    of AST nodes.
    """

    def __init__(self, pattern: str, description: Optional[str] = None):
        self.description = description if description else f"Regex rule: {pattern}"
        try:
            self.pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

    def check(self, node: ast.AST) -> List[SecurityViolation]:
        violations = []

        if isinstance(node, ast.Expr):
            try:
                # Convert node to string representation
                node_str = ast.unparse(node)

                if self.pattern.search(node_str):
                    violations.append(SecurityViolation(
                        message=f"Security rule: {self.description} {f'at line {node.lineno}' if hasattr(node, 'lineno') else ''}"
                    ))
            except Exception:
                # Don't fail analysis due to string conversion issues
                pass

        return violations
