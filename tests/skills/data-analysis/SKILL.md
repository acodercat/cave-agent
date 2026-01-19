---
name: data-analysis
description: Statistical analysis, outlier detection, and data transformation utilities.
license: MIT
compatibility: Python 3.10+
metadata:
  author: test-org
  version: "1.0.0"
---

# Data Analysis Skill

Statistical analysis, outlier detection, and data transformation.

## Quick Start

1. Use `calculate_stats(data)` to compute statistics
2. Use `find_outliers(data, threshold)` to find outliers

## Available Functions

After activating this skill, you have access to:
- `calculate_stats(data: List[float])` - Calculate mean, median, std
- `find_outliers(data: List[float], threshold: float)` - Find outliers using IQR
- `DATA_CONFIG` variable - Configuration dictionary
- `DataPoint` type - Dataclass for structured data
