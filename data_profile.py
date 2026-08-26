from typing import Any, Dict
import pandas as pd


def _is_id_like(col_name: str, series: pd.Series) -> bool:
    """Check if a column is likely an identifier column."""
    name_lower = col_name.lower()
    if any(keyword in name_lower for keyword in ["id", "code", "key", "number"]):
        return True
    return series.nunique() == len(series) and not pd.api.types.is_float_dtype(
        series
    )


def _get_sample_values(series: pd.Series, max_samples: int = 5) -> list:
    """Extract up to max_samples non-null unique values from a series."""
    return series.dropna().unique()[:max_samples].tolist()


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    """Generate statistical summary dictionary for numeric series."""
    clean_series = series.dropna()
    if clean_series.empty:
        return {"min": None, "max": None, "mean": None, "median": None, "std": None}

    return {
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "mean": float(clean_series.mean()),
        "median": float(clean_series.median()),
        "std": float(clean_series.std()) if len(clean_series) > 1 else 0.0,
    }


def _classify_column(series: pd.Series) -> str:
    """Classify a pandas Series into data categories."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_categorical_dtype(series) or series.nunique() < (len(series) * 0.2): # pyright: ignore[reportAttributeAccessIssue]
        return "categorical"
    return "text"


def profile_column(series: pd.Series, col_name: str = "") -> Dict[str, Any]:
    """Profile an individual pandas Series column."""
    col_type = _classify_column(series)
    col_info: Dict[str, Any] = {
        "type": col_type,
        "dtype": str(series.dtype),
        "missing_count": int(series.isnull().sum()),
        "unique_count": int(series.nunique()),
        "is_id_like": _is_id_like(col_name, series),
    }

    if col_type == "numeric":
        col_info.update(_numeric_summary(series))
    elif col_type == "categorical":
        col_info["sample_values"] = _get_sample_values(series)

    return col_info


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset metadata and column profile summary."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    columns_summary = {
        col: profile_column(df[col], col_name=col) for col in df.columns
    }
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_summary,
    }


# Exported function aliases for test suite compatibility
create_profile = profile_data
get_profile = profile_data
_profile_column = profile_column