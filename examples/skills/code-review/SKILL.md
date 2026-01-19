---
name: code-review
description: Review Python code for bugs, security issues, and best practices
---

# Code Review Guidelines

**Important: Do NOT execute the code. Only review and analyze it.**

When reviewing code, follow these steps:

1. **Security Check**: Look for common vulnerabilities
   - SQL injection
   - Command injection
   - Hardcoded credentials

2. **Error Handling**: Verify proper exception handling
   - Are exceptions caught appropriately?
   - Are error messages informative?

3. **Code Quality**: Check for best practices
   - Clear variable names
   - Proper documentation
   - No code duplication

4. **Performance**: Identify potential issues
   - Unnecessary loops
   - Memory leaks
   - Blocking operations

Provide your review in this format:
- **Issues Found**: List each issue with severity (High/Medium/Low)
- **Suggestions**: Actionable improvements
- **Summary**: Overall assessment
