---
name: data-processor
description: Analyze sales data with statistics, regional performance, and commission calculations
---

# Data Processor Skill

Use this skill to analyze the stored sales data.

## Scripts

### process.py
Analyze sales data and compare against regional targets.

Parameters:
- `region` (optional): Filter by region (e.g., "north"). Analyzes all regions if not provided.

Returns: Statistics (count, total, average, min, max) and performance vs target.

### commission.py
Calculate commissions for sales transactions.

Parameters:
- `category` (optional): Filter by category (e.g., "electronics"). Calculates for all if not provided.

Returns: Commission breakdown per transaction and totals.

## Injected Variables

After activating this skill, scripts have access to:

- `sales_data`: List[Dict] of sales transactions with keys: amount, category, region
- `regional_targets`: Dict[str, float] mapping region to sales target
