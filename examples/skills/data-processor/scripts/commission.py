"""
Commission calculation script.

Calculates commissions for the injected sales_data.
"""

from typing import Dict, Any, Optional
from cave_agent.runtime import PythonRuntime

# Commission rates by category
RATES = {
    "electronics": 0.08,
    "clothing": 0.12,
    "furniture": 0.10,
    "groceries": 0.05,
}


def main(
    runtime: PythonRuntime,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate commissions for sales transactions.

    Args:
        runtime: Agent's runtime for retrieving injected data
        category: Optional category to filter by (e.g., "electronics")

    Returns:
        Dictionary with commission breakdown and totals
    """
    # Get injected sales data
    try:
        sales_data = runtime.retrieve("sales_data")
    except KeyError:
        return {"error": "No sales_data available"}

    # Filter by category if specified
    if category:
        sales = [s for s in sales_data if s.get("category") == category]
    else:
        sales = sales_data

    if not sales:
        return {"error": f"No sales found for category: {category}"}

    # Calculate commissions
    results = []
    total_sales = 0
    total_commission = 0

    for sale in sales:
        amount = sale.get("amount", 0)
        cat = sale.get("category", "other")
        rate = RATES.get(cat, 0.05)
        commission = amount * rate

        results.append({
            "amount": amount,
            "category": cat,
            "region": sale.get("region"),
            "rate": f"{rate:.0%}",
            "commission": round(commission, 2),
        })

        total_sales += amount
        total_commission += commission

    return {
        "filter": category or "all categories",
        "transactions": results,
        "total_sales": round(total_sales, 2),
        "total_commission": round(total_commission, 2),
        "effective_rate": f"{total_commission / total_sales:.1%}" if total_sales else "0%",
    }
