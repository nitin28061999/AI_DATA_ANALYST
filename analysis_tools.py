from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


# ============================================================
# DATAFRAME VALIDATION & FILTERING HELPERS
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that input is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")


def normalize_filters(
    filters: List[Dict[str, Any]], 
    df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Validate and normalize filter definitions against DataFrame columns.
    Expected filter structure: {"column": "Region", "operator": "==", "value": "West"}
    """
    if not isinstance(filters, list):
        raise ValueError("Filters must be a list of condition dictionaries.")

    valid_operators = {"==", "!=", ">", "<", ">=", "<=", "in", "not in", "contains"}
    normalized = []

    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op = str(f.get("operator", "==")).lower()
        val = f.get("value")

        if not col or col not in df.columns:
            raise ValueError(f"Filter column '{col}' does not exist in DataFrame.")
        if op not in valid_operators:
            raise ValueError(f"Unsupported filter operator '{op}'.")

        normalized.append({"column": col, "operator": op, "value": val})

    return normalized


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply normalized filter conditions to a DataFrame."""
    filtered_df = df.copy()

    for f in filters:
        col = f["column"]
        op = f["operator"]
        val = f["value"]

        if op == "==":
            filtered_df = filtered_df[filtered_df[col] == val]
        elif op == "!=":
            filtered_df = filtered_df[filtered_df[col] != val]
        elif op == ">":
            filtered_df = filtered_df[filtered_df[col] > val]
        elif op == "<":
            filtered_df = filtered_df[filtered_df[col] < val]
        elif op == ">=":
            filtered_df = filtered_df[filtered_df[col] >= val]
        elif op == "<=":
            filtered_df = filtered_df[filtered_df[col] <= val]
        elif op == "in":
            filtered_df = filtered_df[filtered_df[col].isin(val if isinstance(val, (list, tuple, set)) else [val])]
        elif op == "not in":
            filtered_df = filtered_df[~filtered_df[col].isin(val if isinstance(val, (list, tuple, set)) else [val])]
        elif op == "contains":
            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)]

    return filtered_df


# ============================================================
# UNFILTERED AGGREGATIONS
# ============================================================

def calculate_sum(df: pd.DataFrame, column: str) -> Union[int, float]:
    validate_dataframe(df)
    return float(df[column].sum())


def calculate_average(df: pd.DataFrame, column: str) -> Union[int, float]:
    validate_dataframe(df)
    return float(df[column].mean())


def calculate_count(df: pd.DataFrame, column: Optional[str] = None) -> int:
    validate_dataframe(df)
    return int(df[column].count()) if column else len(df)


def calculate_unique_count(df: pd.DataFrame, column: str) -> int:
    validate_dataframe(df)
    return int(df[column].nunique())


def calculate_min(df: pd.DataFrame, column: str) -> Any:
    validate_dataframe(df)
    val = df[column].min()
    return float(val) if isinstance(val, (int, float)) else str(val)


def calculate_max(df: pd.DataFrame, column: str) -> Any:
    validate_dataframe(df)
    val = df[column].max()
    return float(val) if isinstance(val, (int, float)) else str(val)


def group_and_sum(df: pd.DataFrame, group_by: str, column: str) -> pd.DataFrame:
    validate_dataframe(df)
    return df.groupby(group_by, as_index=False)[column].sum() # pyright: ignore[reportReturnType]


def group_and_average(df: pd.DataFrame, group_by: str, column: str) -> pd.DataFrame:
    validate_dataframe(df)
    return df.groupby(group_by, as_index=False)[column].mean() # pyright: ignore[reportReturnType]


def group_and_count(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    validate_dataframe(df)
    return df.groupby(group_by, as_index=False).size().rename(columns={"size": "count"})


def value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    validate_dataframe(df)
    return df[column].value_counts().reset_index().rename(columns={"index": column, column: "count"})


def top_n(df: pd.DataFrame, group_by: str, column: str, n: int) -> pd.DataFrame:
    validate_dataframe(df)
    grouped = df.groupby(group_by, as_index=False)[column].sum()
    return grouped.sort_values(by=column, ascending=False).head(n).reset_index(drop=True) # pyright: ignore[reportCallIssue]


# ============================================================
# FILTERED AGGREGATIONS
# ============================================================

def filtered_sum(df: pd.DataFrame, filters: List[Dict[str, Any]], column: str) -> Union[int, float]:
    filtered = apply_filters(df, filters)
    return float(filtered[column].sum()) if not filtered.empty else 0.0


def filtered_average(df: pd.DataFrame, filters: List[Dict[str, Any]], column: str) -> Union[int, float]:
    filtered = apply_filters(df, filters)
    return float(filtered[column].mean()) if not filtered.empty else 0.0


def filtered_count(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> int:
    filtered = apply_filters(df, filters)
    return len(filtered)


def filtered_unique_count(df: pd.DataFrame, filters: List[Dict[str, Any]], column: str) -> int:
    filtered = apply_filters(df, filters)
    return int(filtered[column].nunique()) if not filtered.empty else 0


def filtered_min(df: pd.DataFrame, filters: List[Dict[str, Any]], column: str) -> Any:
    filtered = apply_filters(df, filters)
    if filtered.empty:
        return None
    val = filtered[column].min()
    return float(val) if isinstance(val, (int, float)) else str(val)


def filtered_max(df: pd.DataFrame, filters: List[Dict[str, Any]], column: str) -> Any:
    filtered = apply_filters(df, filters)
    if filtered.empty:
        return None
    val = filtered[column].max()
    return float(val) if isinstance(val, (int, float)) else str(val)


def filtered_group_and_sum(df: pd.DataFrame, filters: List[Dict[str, Any]], group_by: str, column: str) -> pd.DataFrame:
    filtered = apply_filters(df, filters)
    return group_and_sum(filtered, group_by, column) if not filtered.empty else pd.DataFrame(columns=[group_by, column])


def filtered_group_and_average(df: pd.DataFrame, filters: List[Dict[str, Any]], group_by: str, column: str) -> pd.DataFrame:
    filtered = apply_filters(df, filters)
    return group_and_average(filtered, group_by, column) if not filtered.empty else pd.DataFrame(columns=[group_by, column])


def filtered_value_counts(df: pd.DataFrame, filters: List[Dict[str, Any]], column: str) -> pd.DataFrame:
    filtered = apply_filters(df, filters)
    return value_counts(filtered, column) if not filtered.empty else pd.DataFrame(columns=[column, "count"])


def filtered_top_n(df: pd.DataFrame, filters: List[Dict[str, Any]], group_by: str, column: str, n: int) -> pd.DataFrame:
    filtered = apply_filters(df, filters)
    return top_n(filtered, group_by, column, n) if not filtered.empty else pd.DataFrame(columns=[group_by, column])