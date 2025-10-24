import pytest

from cave_agent.security_checker import (
    SecurityChecker, SecurityError, ImportRule, FunctionRule, AttributeRule, RegexRule
)
from cave_agent.python_runtime import PythonRuntime, PythonExecutor


class TestSecurityChecker:
    """Test suite for SecurityChecker core functionality."""
    
    def test_security_checker_initialization(self):
        """Test SecurityChecker initialization with rules."""
        # Test initialization with rules
        rules = [
            ImportRule({"os", "subprocess"}),
            FunctionRule({"eval", "exec"})
        ]
        checker = SecurityChecker(rules)
        assert len(checker.rules) == 2
        
        # Test empty initialization
        empty_checker = SecurityChecker([])
        assert len(empty_checker.rules) == 0
    
    def test_add_rule(self):
        """Test adding rules to SecurityChecker."""
        checker = SecurityChecker([])
        
        rule = ImportRule({"os"})
        checker.add_rule(rule)
        assert len(checker.rules) == 1
        
        # Test duplicate rule name should raise error
        duplicate_rule = ImportRule({"sys"})  # Same name "forbidden_imports"
        with pytest.raises(ValueError):
            checker.add_rule(duplicate_rule)
    
    def test_import_rule(self):
        """Test ImportRule detection."""
        rule = ImportRule({"os", "subprocess"})
        checker = SecurityChecker([rule])
        
        # Test forbidden imports
        forbidden_codes = [
            "import os",
            "import subprocess", 
            "from os import system",
            "from subprocess import call"
        ]
        
        for code in forbidden_codes:
            violations = checker.check_code(code)
            assert len(violations) > 0
            assert any("import" in v.message.lower() for v in violations)
        
        # Test safe import
        safe_code = "import math"
        violations = checker.check_code(safe_code)
        assert len(violations) == 0
    
    def test_function_rule(self):
        """Test FunctionRule detection."""
        rule = FunctionRule({"eval", "exec", "compile"})
        checker = SecurityChecker([rule])
        
        # Test forbidden functions
        forbidden_codes = [
            "eval('print(1)')",
            "exec('x = 1')",
            "compile('x=1', '<string>', 'exec')"
        ]
        
        for code in forbidden_codes:
            violations = checker.check_code(code)
            assert len(violations) > 0
            assert any("function" in v.message.lower() for v in violations)
        
        # Test safe function
        safe_code = "print('hello')"
        violations = checker.check_code(safe_code)
        assert len(violations) == 0
    
    def test_attribute_rule(self):
        """Test AttributeRule detection."""
        rule = AttributeRule({"__globals__", "__builtins__"})
        checker = SecurityChecker([rule])
        
        # Test forbidden attribute access
        forbidden_codes = [
            "print.__globals__",
            "obj.__builtins__"
        ]
        
        for code in forbidden_codes:
            violations = checker.check_code(code)
            assert len(violations) > 0
            assert any("attribute" in v.message.lower() for v in violations)
        
        # Test safe attribute access
        safe_code = "obj.method()"
        violations = checker.check_code(safe_code)
        assert len(violations) == 0
    
    def test_regex_rule(self):
        """Test RegexRule functionality."""
        rule = RegexRule("no_print", "Block print statements", r"print\s*\(")
        checker = SecurityChecker([rule])
        
        # Test pattern matching
        code = "print('hello world')"
        violations = checker.check_code(code)
        assert len(violations) > 0
        assert "no_print" in violations[0].message
        
        # Test safe code
        safe_code = "x = 5 + 3"
        violations = checker.check_code(safe_code)
        assert len(violations) == 0
    
    def test_regex_rule_validation(self):
        """Test RegexRule parameter validation."""
        # Test invalid regex pattern
        with pytest.raises(ValueError):
            RegexRule("bad_pattern", "Bad regex", "[unclosed")
    
    def test_multiple_rules(self):
        """Test SecurityChecker with multiple rules."""
        rules = [
            ImportRule({"os"}),
            FunctionRule({"eval"}),
            AttributeRule({"__globals__"}),
            RegexRule("no_print", "Block print statements", r"print\s*\(")
        ]
        checker = SecurityChecker(rules)
        
        # Test code that violates multiple rules
        complex_code = """
import os
eval('print(1)')
obj.__globals__
print('test')
"""
        violations = checker.check_code(complex_code)
        assert len(violations) >= 4  # Should catch import, eval, globals, and print
    
    def test_syntax_error_handling(self):
        """Test handling of syntax errors in code."""
        checker = SecurityChecker([ImportRule({"os"})])
        
        # Code with syntax error
        bad_code = "def test(\nprint('missing parenthesis')"
        violations = checker.check_code(bad_code)
        assert len(violations) >= 1
        assert any("syntax error" in v.message.lower() for v in violations)
    
    def test_empty_code_handling(self):
        """Test handling of empty code."""
        checker = SecurityChecker([ImportRule({"os"})])
        
        # Test empty code
        violations = checker.check_code("")
        assert len(violations) >= 1
        assert any("empty" in v.message.lower() for v in violations)
        
        violations = checker.check_code("   ")
        assert len(violations) >= 1
        assert any("empty" in v.message.lower() for v in violations)
    
    def test_safe_code_passes(self):
        """Test that safe code passes security checks."""
        rules = [
            ImportRule({"os", "subprocess"}),
            FunctionRule({"eval", "exec"}),
            AttributeRule({"__globals__"})
        ]
        checker = SecurityChecker(rules)
        
        safe_codes = [
            "x = 5 + 3",
            "def hello(): return 'world'",
            "[i**2 for i in range(10)]",
            "{'key': 'value'}",
            "import math; math.sqrt(16)"
        ]
        
        for code in safe_codes:
            violations = checker.check_code(code)
            # Should have no violations for safe code
            assert len(violations) == 0, f"Safe code raised violations: {code} -> {violations}"


class TestPythonRuntimeSecurity:
    """Test suite for PythonRuntime security integration."""
    
    @pytest.mark.asyncio
    async def test_runtime_with_security_enabled(self):
        """Test PythonRuntime with security checking enabled."""
        rules = [
            ImportRule({"os"}),
            FunctionRule({"eval"})
        ]
        checker = SecurityChecker(rules)
        runtime = PythonRuntime(security_checker=checker)
        
        # Test safe code execution
        safe_code = "x = 5 + 3"
        result = await runtime.execute(safe_code)
        assert result.success
        
        # Test forbidden code execution
        forbidden_code = "import os; os.system('ls')"
        result = await runtime.execute(forbidden_code)
        assert not result.success
        assert isinstance(result.error, SecurityError)
        assert "violations" in str(result.error).lower()
    
    @pytest.mark.asyncio
    async def test_runtime_with_security_disabled(self):
        """Test PythonRuntime with security checking disabled."""
        runtime = PythonRuntime(security_checker=None)
        
        # Any code should execute when security is disabled
        code = "result = 'security_disabled'"
        result = await runtime.execute(code)
        assert result.success
    
    @pytest.mark.asyncio
    async def test_runtime_regex_rule_security_rules(self):
        """Test adding regex security rules to runtime."""
        # RegexRule only works on expressions, so test with a print statement
        regex_rule = RegexRule("no_print", "Disallow print statements", r"print\s*\(")
        checker = SecurityChecker([regex_rule])
        runtime = PythonRuntime(security_checker=checker)
        
        # Test that regex rule works
        print_code = "print('hello world')"
        result = await runtime.execute(print_code)
        assert not result.success


class TestPythonExecutorSecurity:
    """Test suite for PythonExecutor security integration."""
    
    @pytest.mark.asyncio
    async def test_executor_security_integration(self):
        """Test PythonExecutor with SecurityChecker."""
        rules = [ImportRule({"os"}), FunctionRule({"eval"})]
        checker = SecurityChecker(rules)
        executor = PythonExecutor(security_checker=checker)
        
        # Test safe code
        safe_code = "x = 42"
        result = await executor.execute(safe_code)
        assert result.success
        
        # Test forbidden code
        forbidden_code = "eval('print(1)')"
        result = await executor.execute(forbidden_code)
        assert not result.success
        assert isinstance(result.error, SecurityError)
        assert "violations" in result.error.message.lower()
    
    @pytest.mark.asyncio
    async def test_executor_without_security(self):
        """Test PythonExecutor without security checking."""
        executor = PythonExecutor(security_checker=None)
        
        # Any code should execute (subject to Python's own restrictions)
        code = "x = 'no_security'"
        result = await executor.execute(code)
        assert result.success


def test_security_error_exception():
    """Test SecurityError exception handling."""
    error = SecurityError("Test security error")
    assert "Test security error" in str(error)


class TestSecurityIntegration:
    """Test integration scenarios and edge cases."""
    
    def test_comprehensive_security_setup(self):
        """Test a comprehensive security setup."""
        rules = [
            ImportRule({"os", "subprocess", "sys", "shutil", "socket", "urllib"}),
            FunctionRule({"eval", "exec", "compile", "open", "__import__", "globals", "locals"}),
            AttributeRule({"__globals__", "__locals__", "__code__", "__builtins__"}),
            RegexRule("no_print", "Block print statements", r"print\s*\(")
        ]
        checker = SecurityChecker(rules)
        
        # Test various forbidden operations
        forbidden_operations = [
            "import os",
            "eval('1+1')",  
            "obj.__globals__",
            "print('test')",  # RegexRule only works on expressions
            "open('file.txt')"
        ]
        
        for operation in forbidden_operations:
            violations = checker.check_code(operation)
            assert len(violations) > 0, f"Should detect violation in: {operation}"
    
    def test_complex_code_analysis(self):
        """Test analysis of complex code structures."""
        rules = [ImportRule({"os"}), FunctionRule({"eval"})]
        checker = SecurityChecker(rules)
        
        complex_code = """
class MyClass:
    def __init__(self):
        self.data = []
    
    def process(self, items):
        for item in items:
            if hasattr(item, 'value'):
                self.data.append(item.value)
        return len(self.data)

def main():
    obj = MyClass()
    return obj.process([1, 2, 3])

if __name__ == "__main__":
    main()
"""
        
        violations = checker.check_code(complex_code)
        # This should be safe code with no violations
        assert len(violations) == 0