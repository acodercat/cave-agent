"""
Data transformation script.

Provides utilities for transforming and cleaning data.
"""
from typing import List, Any, Callable, Optional


def main(
    runtime,
    data: List[float],
    operation: str = "normalize",
    **kwargs
) -> List[float]:
    """
    Transform data using specified operation.

    Args:
        runtime: Agent's PythonRuntime
        data: List of numeric values
        operation: One of "normalize", "standardize", "scale", "clip"
        **kwargs: Operation-specific parameters

    Returns:
        Transformed data list
    """
    if not data:
        return []

    operations = {
        "normalize": _normalize,
        "standardize": _standardize,
        "scale": _scale,
        "clip": _clip,
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}. Available: {list(operations.keys())}")

    return operations[operation](data, **kwargs)


def _normalize(data: List[float], **kwargs) -> List[float]:
    """Normalize to [0, 1] range."""
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val

    if range_val == 0:
        return [0.5] * len(data)

    return [(x - min_val) / range_val for x in data]


def _standardize(data: List[float], **kwargs) -> List[float]:
    """Standardize to zero mean and unit variance."""
    import statistics

    mean = statistics.mean(data)
    stdev = statistics.stdev(data) if len(data) > 1 else 1

    if stdev == 0:
        return [0.0] * len(data)

    return [(x - mean) / stdev for x in data]


def _scale(data: List[float], factor: float = 1.0, **kwargs) -> List[float]:
    """Scale by a factor."""
    return [x * factor for x in data]


def _clip(data: List[float], min_val: float = 0.0, max_val: float = 1.0, **kwargs) -> List[float]:
    """Clip values to [min_val, max_val] range."""
    return [max(min_val, min(max_val, x)) for x in data]
