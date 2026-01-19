# API Reference

## Functions

### calculate_stats(data: List[float]) -> Dict[str, float]

Calculate basic statistics for a dataset.

**Parameters:**
- `data`: List of numeric values

**Returns:**
Dictionary containing:
- `mean`: Arithmetic mean
- `median`: Median value
- `stdev`: Standard deviation (0 if single value)
- `min`: Minimum value
- `max`: Maximum value

**Example:**
```python
calculate_stats([10, 20, 30, 40, 50])
# {'mean': 30.0, 'median': 30, 'stdev': 15.81, 'min': 10, 'max': 50}
```

---

### find_outliers(data: List[float], threshold: float = 1.5) -> List[float]

Find outliers using the Interquartile Range (IQR) method.

**Parameters:**
- `data`: List of numeric values
- `threshold`: IQR multiplier (default: 1.5)

**Returns:**
List of values considered outliers.

**Example:**
```python
find_outliers([1, 2, 3, 4, 5, 100])
# [100]
```

## Variables

### DATA_CONFIG

Default configuration dictionary.

```python
{
    "default_threshold": 1.5,
    "max_data_points": 10000,
    "supported_formats": ["csv", "json", "parquet"],
}
```

## Types

### DataPoint

Dataclass for structured data points.

**Fields:**
- `value: float` - The numeric value
- `label: str` - Optional label (default: "")
- `timestamp: str` - Optional timestamp (default: "")

**Methods:**
- `to_dict() -> Dict[str, Any]` - Convert to dictionary

### AnalysisResult

Dataclass for analysis results.

**Fields:**
- `stats: Dict[str, float]` - Statistics dictionary
- `outliers: List[float]` - List of outlier values
- `data_count: int` - Number of data points analyzed

**Properties:**
- `has_outliers: bool` - True if outliers were found
