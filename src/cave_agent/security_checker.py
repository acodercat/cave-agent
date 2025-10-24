import ast
import re
from abc import ABC, abstractmethod
from typing import List, Set
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
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def check(self, node: ast.AST) -> List[SecurityViolation]:
        """Check if the AST node violates this rule.
        
        Args:
            node: AST node to analyze
            
        Returns:
            List of violations found (empty if none)
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ImportRule(SecurityRule):
    """Rule to detect forbidden imports."""
    
    def __init__(self, forbidden_modules: Set[str]):
        super().__init__(
            "forbidden_imports",
            "Detects forbidden imports"
        )
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
    
    def __init__(self, forbidden_functions: Set[str]):
        super().__init__(
            "forbidden_functions",
            "Detects forbidden function calls"
        )
        self.forbidden_functions = forbidden_functions
    
    def check(self, node: ast.AST) -> List[SecurityViolation]:
        violations = []
        
        if isinstance(node, ast.Call):
            func_name = self._get_function_name(node.func)
            if func_name in self.forbidden_functions:
                violations.append(SecurityViolation(
                    message=f"Forbidden function call detected: {func_name} at line {node.lineno}",
                ))
        
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
        super().__init__(
            "forbidden_attributes",
            "Detects forbidden attribute access"
        )
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
    
    def __init__(self, name: str, description: str, pattern: str):
        super().__init__(name, description)
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
                        message=f"Security rule '{self.name}': {self.description} {f'at line {node.lineno}' if hasattr(node, 'lineno') else ''}"
                    ))
            except Exception:
                # Don't fail analysis due to string conversion issues
                pass
        
        return violations


class SecurityChecker:
    """Main security checker for Python code analysis.
    
    Provides comprehensive security analysis using AST parsing to detect
    security violations before code execution. Supports multiple
    security rules.
    
    Example:
        >>> from cave_agent.security_checker import SecurityChecker, ImportRule, FunctionRule, AttributeRule, EvalExecRule, RegexRule
        >>> checker = SecurityChecker([
        >>>     ImportRule(set(["os", "subprocess", "sys", "shutil", "pathlib", "socket", "urllib", "http", "ctypes", "gc", "csv"])),
        >>>     FunctionRule(set(["eval", "exec", "compile", "open", "input", "raw_input", "exit", "quit", "__import__", "globals", "locals", "breakpoint"])),
        >>>     AttributeRule(set(["__globals__", "__locals__", "__code__", "__closure__", "__defaults__", "__dict__", "__class__", "__bases__", "__mro__", "__subclasses__", "__import__", "__builtins__"])),
        >>>     RegexRule("forbidden_statements", "Detects forbidden statements", r"delete")
        >>> ])
        >>> violations = checker.check_code("import os; os.system('ls')")
        >>> print(len(violations))
    """
    
    def __init__(
        self,
        rules: List[SecurityRule]
    ):
        """Initialize SecurityChecker with specified rules.
        
        Args:
            rules: List of security rules
            
        Raises:
            ValueError: If rule with same name already exists
        """
        self.rules = []
        for rule in rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: SecurityRule):
        """Add a security rule.
        
        Args:
            rule: Security rule to add
            
        Raises:
            ValueError: If rule with same name already exists
        """
        if any(r.name == rule.name for r in self.rules):
            raise ValueError(f"Rule with name '{rule.name}' already exists")
            
        self.rules.append(rule)
    
    def check_code(self, code: str) -> List[SecurityViolation]:
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
            violations.append(SecurityViolation(
                message="Parse error: Code cannot be empty",
            ))
            
        try:
            # Parse code into AST
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(SecurityViolation(
                message=f"Syntax error: {str(e)}",
            ))
            return violations
        except Exception as e:
            violations.append(SecurityViolation(
                message=f"Parse error: {str(e)}",
            ))
            return violations
        
        # Analyze AST with all rules
        for node in ast.walk(tree):
            for rule in self.rules:
                try:
                    violations.extend(rule.check(node))
                except Exception:
                    # Don't let rule failures break analysis
                    continue
        
        return violations

class SecurityError(Exception):
    """Exception raised when code fails security checks."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
    