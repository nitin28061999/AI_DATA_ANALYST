from typing import Any, Dict, List
import pandas as pd
import numpy as np

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the input is a valid, non-empty pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")

def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive profiling metadata for a pandas DataFrame.
    
    Returns structured stats on dimensions, memory usage, missing values,
    data types, numeric summaries, and sample distinct values.
    """
    validate_dataframe(df)

    total_rows, total_cols = df.shape
    columns_info: Dict[str, Dict[str, Any]] = {}

    for column in df.columns:
        col_data = df[column]
        dtype_str = str(col_data.dtype)
        null_count = int(col_data.isnull().sum())
        null_percentage = round((null_count / total_rows) * 100, 2)
        unique_count = int(col_data.nunique(dropna=True))

        info: Dict[str, Any] = {
            "dtype": dtype_str,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "unique_count": unique_count,
        }

        # Numeric column summaries
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            non_null = col_data.dropna()
            if not non_null.empty:
                info.update({
                    "min": float(non_null.min()) if not np.isinf(non_null.min()) else None,
                    "max": float(non_null.max()) if not np.isinf(non_null.max()) else None,
                    "mean": round(float(non_null.mean()), 4),
                    "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0,
                    "median": float(non_null.median()),
                })

        # Categorical / String column sample values
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            distinct_samples: List[Any] = col_data.dropna().unique()[:5].tolist()
            info["sample_values"] = distinct_samples

        columns_info[str(column)] = info

    return {
        "summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        },
        "columns": columns_info,
    }
