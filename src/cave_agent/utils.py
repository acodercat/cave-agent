import re
from typing import Union

def extract_python_code(response, python_block_identifier: str) -> Union[str, None]:
    """Extract python code block from LLM output"""
    results = []
    lines = response.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        # Check if this line starts a code block
        if line.strip().lower() == f'```{python_block_identifier.lower()}':
            # Start collecting code
            code_lines = []
            i += 1
            
            # Collect lines until we hit closing ``` or end of text
            while i < len(lines):
                current_line = lines[i]
                # Check if this is a closing ```
                if current_line.strip() == '```':
                    # This is the end of the code block
                    break
                code_lines.append(current_line)
                i += 1
            
            # Add the collected code (even if no closing ``` was found)
            if code_lines:
                code = '\n'.join(code_lines).rstrip()
                if code:
                    results.append(code)
        i += 1
    
    # Return joined results or None
    if results:
        return '\n\n'.join(results)
    return None
