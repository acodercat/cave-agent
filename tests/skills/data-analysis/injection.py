"""
Injection module for data-analysis skill.

Exports functions, variables, and types that get injected into the agent's runtime
when the skill is activated.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import statistics

from cave_agent.runtime import Function, Variable, Type


# =============================================================================
# Functions
# =============================================================================

def calculate_stats(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a dataset.

    Args:
        data: List of numeric values

    Returns:
        Dictionary with mean, median, stdev, min, max
    """
    if not data:
        return {"mean": 0, "median": 0, "stdev": 0, "min": 0, "max": 0}

    return {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0,
        "min": min(data),
        "max": max(data),
    }


def find_outliers(data: List[float], threshold: float = 1.5) -> List[float]:
    """
    Find outliers using the IQR method.

    Args:
        data: List of numeric values
        threshold: IQR multiplier (default 1.5)

    Returns:
        List of outlier values
    """
    if len(data) < 4:
        return []

    sorted_data = sorted(data)
    q1 = statistics.median(sorted_data[:len(sorted_data)//2])
    q3 = statistics.median(sorted_data[len(sorted_data)//2:])
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    return [x for x in data if x < lower_bound or x > upper_bound]


# =============================================================================
# Variables
# =============================================================================

DATA_CONFIG = {
    "default_threshold": 1.5,
    "max_data_points": 10000,
    "supported_formats": ["csv", "json", "parquet"],
}


# =============================================================================
# Types
# =============================================================================

@dataclass
class DataPoint:
    """Represents a single data point with metadata."""
    value: float
    label: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "timestamp": self.timestamp,
        }


@dataclass
class AnalysisResult:
    """Result from data analysis."""
    stats: Dict[str, float]
    outliers: List[float]
    data_count: int

    @property
    def has_outliers(self) -> bool:
        return len(self.outliers) > 0


# =============================================================================
# Exports
# =============================================================================

__exports__ = [
    # Functions
    Function(calculate_stats, description="Calculate basic statistics (mean, median, stdev, min, max)"),
    Function(find_outliers, description="Find outliers using IQR method"),

    # Variables
    Variable("DATA_CONFIG", value=DATA_CONFIG, description="Default configuration for data analysis"),

    # Types
    Type(DataPoint, description="A single data point with value, label, and timestamp"),
    Type(AnalysisResult, description="Result from data analysis with stats and outliers"),
]
