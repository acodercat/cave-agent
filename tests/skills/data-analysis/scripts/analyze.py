"""
Data analysis script.

This script performs comprehensive data analysis including statistics,
outlier detection, and trend analysis.
"""
import statistics
from typing import List, Dict, Any


def main(runtime, data: List[float], threshold: float = 1.5) -> Dict[str, Any]:
    """
    Analyze data and return comprehensive results.

    Args:
        runtime: Agent's PythonRuntime for accessing state
        data: List of numeric values to analyze
        threshold: IQR multiplier for outlier detection

    Returns:
        Dictionary with analysis results
    """
    if not data:
        return {
            "error": "No data provided",
            "stats": None,
            "outliers": [],
            "summary": "Empty dataset"
        }

    # Calculate basic statistics
    stats = {
        "count": len(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0,
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data),
    }

    # Find outliers using IQR method
    outliers = []
    if len(data) >= 4:
        sorted_data = sorted(data)
        mid = len(sorted_data) // 2
        q1 = statistics.median(sorted_data[:mid])
        q3 = statistics.median(sorted_data[mid:])
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = [x for x in data if x < lower_bound or x > upper_bound]

    # Generate summary
    summary_parts = [
        f"Analyzed {stats['count']} data points.",
        f"Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}",
    ]
    if outliers:
        summary_parts.append(f"Found {len(outliers)} outliers: {outliers}")
    else:
        summary_parts.append("No outliers detected.")

    return {
        "stats": stats,
        "outliers": outliers,
        "threshold": threshold,
        "summary": " ".join(summary_parts),
    }
