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
2. Use `run_skill_script("data-analysis", "analyze.py", data=[...])` to run analysis
3. Use `read_skill_reference("data-analysis", "GUIDE.md")` for detailed guide
4. Use `read_skill_asset("data-analysis", "config.json")` for configuration

## Available Functions

After activating this skill, you have access to:
- `calculate_stats(data: List[float])` - Calculate mean, median, std
- `find_outliers(data: List[float], threshold: float)` - Find outliers using IQR
- `DATA_CONFIG` variable - Configuration dictionary
- `DataPoint` type - Dataclass for structured data

## Scripts

- `analyze.py` - Comprehensive data analysis
- `validate.py` - Data validation
- `transform.py` - Data transformation utilities

## References

- `GUIDE.md` - Comprehensive usage guide with examples
- `API.md` - API reference documentation

## Assets

- `config.json` - Configuration file with default settings
- `sample-data.csv` - Sample CSV data for testing
