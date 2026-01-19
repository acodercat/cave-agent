"""
Data validation script.

Validates data against configurable rules.
"""
from typing import List, Dict, Any, Optional


def main(
    runtime,
    data: List[Any],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    required_count: Optional[int] = None,
    allow_none: bool = False,
) -> Dict[str, Any]:
    """
    Validate data against rules.

    Args:
        runtime: Agent's PythonRuntime
        data: List of values to validate
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        required_count: Required number of items (optional)
        allow_none: Whether None values are allowed

    Returns:
        Validation result with is_valid flag and errors list
    """
    errors = []
    warnings = []

    # Check required count
    if required_count is not None and len(data) != required_count:
        errors.append(f"Expected {required_count} items, got {len(data)}")

    # Check for None values
    none_count = sum(1 for x in data if x is None)
    if none_count > 0 and not allow_none:
        errors.append(f"Found {none_count} None values (not allowed)")

    # Check numeric values
    numeric_data = [x for x in data if isinstance(x, (int, float)) and x is not None]

    if min_value is not None:
        below_min = [x for x in numeric_data if x < min_value]
        if below_min:
            errors.append(f"Found {len(below_min)} values below minimum {min_value}: {below_min[:5]}")

    if max_value is not None:
        above_max = [x for x in numeric_data if x > max_value]
        if above_max:
            errors.append(f"Found {len(above_max)} values above maximum {max_value}: {above_max[:5]}")

    # Check for non-numeric values (warning only)
    non_numeric = len(data) - none_count - len(numeric_data)
    if non_numeric > 0:
        warnings.append(f"Found {non_numeric} non-numeric values")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checked_count": len(data),
        "numeric_count": len(numeric_data),
    }
