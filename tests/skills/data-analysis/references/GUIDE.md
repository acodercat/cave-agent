# Data Analysis Skill Guide

This is a comprehensive guide for using the data-analysis skill.

## Overview

This skill provides data analysis capabilities including:
- Statistical calculations
- Outlier detection
- Data transformation
- Validation

## Usage Patterns

### Basic Statistics

```python
# After activating the skill
stats = calculate_stats([1, 2, 3, 4, 5, 100])
print(stats)
# {'mean': 19.17, 'median': 3.5, 'stdev': 39.4, 'min': 1, 'max': 100}
```

### Finding Outliers

```python
outliers = find_outliers([1, 2, 3, 4, 5, 100], threshold=1.5)
print(outliers)
# [100]
```

### Using DataPoint Type

```python
point = DataPoint(value=42.5, label="measurement", timestamp="2024-01-15")
print(point.to_dict())
```

## Best Practices

1. Always validate data before analysis
2. Use appropriate threshold for outlier detection
3. Consider data distribution when choosing transformations
4. Document any preprocessing steps

## Troubleshooting

### Common Issues

- **Empty results**: Ensure data list is not empty
- **Outlier threshold**: Adjust threshold based on data distribution
- **Type errors**: Ensure all values are numeric

## API Reference

See the SKILL.md for function signatures and descriptions.
