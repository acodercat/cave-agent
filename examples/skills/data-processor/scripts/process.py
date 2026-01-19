"""
Process sales data script.

Analyzes the injected sales_data and compares against regional_targets.
"""

from typing import Dict, Any, Optional
from cave_agent.runtime import PythonRuntime


def main(
    runtime: PythonRuntime,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze sales data, optionally filtered by region.

    Args:
        runtime: Agent's runtime for retrieving injected data
        region: Optional region to filter by (e.g., "north")

    Returns:
        Dictionary with sales statistics and performance vs target
    """
    # Get injected data
    try:
        sales_data = runtime.retrieve("sales_data")
    except KeyError:
        return {"error": "No sales_data available"}

    try:
        targets = runtime.retrieve("regional_targets")
    except KeyError:
        targets = {}

    # Filter by region if specified
    if region:
        sales = [s for s in sales_data if s.get("region") == region]
        target = targets.get(region)
    else:
        sales = sales_data
        target = sum(targets.values()) if targets else None

    if not sales:
        return {"error": f"No sales found for region: {region}"}

    # Calculate statistics
    amounts = [s["amount"] for s in sales]
    total = sum(amounts)

    result = {
        "region": region or "all",
        "transaction_count": len(sales),
        "total_sales": total,
        "average_sale": round(total / len(sales), 2),
        "min_sale": min(amounts),
        "max_sale": max(amounts),
    }

    # Add performance vs target if available
    if target:
        variance = total - target
        result["target"] = target
        result["variance"] = variance
        result["performance"] = f"{variance:+.0f} ({variance/target:+.1%})"

    return result
