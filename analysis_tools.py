from typing import Any, Dict, List, Optional, Union
import pandas as pd


def validate_dataframe(df: Any) -> pd.DataFrame:
    """Validate that the input is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")
    return df


# Maintain backwards compatibility for internal module calls
_validate_df = validate_dataframe


def normalize_filters(df: pd.DataFrame, filters: Any) -> List[Dict[str, Any]]:
    if filters is None:
        return []
    if isinstance(filters, dict):
        filters = [filters]
    if not isinstance(filters, list):
        raise ValueError("Filters must be a list of condition dictionaries.")

    normalized = []
    op_map = {
        "=": "==",
        "==": "==",
        "===": "==",
        "====": "==",
        "eq": "==",
        "equals": "==",
        "equal": "==",
        "!=": "!=",
        "ne": "!=",
        "not equal": "!=",
        "not_equal": "!=",
        ">": ">",
        "gt": ">",
        "greater than": ">",
        "greater_than": ">",
        ">=": ">=",
        "gte": ">=",
        "greater than or equal": ">=",
        "greater_than_or_equal": ">=",
        "<": "<",
        "lt": "<",
        "less than": "<",
        "less_than": "<",
        "<=": "<=",
        "lte": "<=",
        "less than or equal": "<=",
        "less_than_or_equal": "<=",
    }

    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        if not col or col not in df.columns:
            continue
        op = str(f.get("operator", "==")).strip().lower()
        if op not in op_map:
            raise ValueError(f"Invalid operator: {op}")

        val = f.get("value")
        normalized.append({"column": col, "operator": op_map[op], "value": val})

    return normalized


def apply_filters(df: pd.DataFrame, filters: Any) -> pd.DataFrame:
    df = validate_dataframe(df)
    norm_filters = normalize_filters(df, filters)
    filtered_df = df.copy()

    for f in norm_filters:
        col = f["column"]
        op = f["operator"]
        val = f["value"]

        if op == "==":
            filtered_df = filtered_df[filtered_df[col] == val]
        elif op == "!=":
            filtered_df = filtered_df[filtered_df[col] != val]
        elif op == ">":
            filtered_df = filtered_df[filtered_df[col] > val]
        elif op == ">=":
            filtered_df = filtered_df[filtered_df[col] >= val]
        elif op == "<":
            filtered_df = filtered_df[filtered_df[col] < val]
        elif op == "<=":
            filtered_df = filtered_df[filtered_df[col] <= val]

    return filtered_df


# Aggregations
def calculate_sum(df: pd.DataFrame, column: str) -> float:
    df = validate_dataframe(df)
    return float(pd.to_numeric(df[column], errors="coerce").sum())


def calculate_average(df: pd.DataFrame, column: str) -> float:
    df = validate_dataframe(df)
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def calculate_count(df: pd.DataFrame, column: Optional[str] = None) -> int:
    df = validate_dataframe(df)
    if column:
        return int(df[column].count())
    return int(len(df))


def calculate_unique_count(df: pd.DataFrame, column: str) -> int:
    df = validate_dataframe(df)
    return int(df[column].nunique())


def calculate_min(df: pd.DataFrame, column: str) -> float:
    df = validate_dataframe(df)
    return float(pd.to_numeric(df[column], errors="coerce").min())


def calculate_max(df: pd.DataFrame, column: str) -> float:
    df = validate_dataframe(df)
    return float(pd.to_numeric(df[column], errors="coerce").max())


# Filtered Aggregations
def filtered_sum(df: pd.DataFrame, filters: Any, column: str) -> float:
    f_df = apply_filters(df, filters)
    return calculate_sum(f_df, column)


def filtered_average(df: pd.DataFrame, filters: Any, column: str) -> float:
    f_df = apply_filters(df, filters)
    return calculate_average(f_df, column)


def filtered_count(df: pd.DataFrame, filters: Any, column: Optional[str] = None) -> int:
    f_df = apply_filters(df, filters)
    return calculate_count(f_df, column)


def filtered_unique_count(df: pd.DataFrame, filters: Any, column: str) -> int:
    f_df = apply_filters(df, filters)
    return calculate_unique_count(f_df, column)


def filtered_min(df: pd.DataFrame, filters: Any, column: str) -> float:
    f_df = apply_filters(df, filters)
    return calculate_min(f_df, column)


def filtered_max(df: pd.DataFrame, filters: Any, column: str) -> float:
    f_df = apply_filters(df, filters)
    return calculate_max(f_df, column)


# Grouped & Value Counts
def group_and_sum(df: pd.DataFrame, group_by: str, column: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    res = df.groupby(group_by)[column].sum().reset_index()
    res.columns = [group_by, column]
    return res


def group_and_average(df: pd.DataFrame, group_by: str, column: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    res = df.groupby(group_by)[column].mean().reset_index()
    res.columns = [group_by, column]
    return res


def group_and_count(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    res = df.groupby(group_by).size().reset_index(name="Count")
    return res


def value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    res = df[column].value_counts().reset_index()
    res.columns = [column, "Count"]
    return res


def filtered_group_and_sum(df: pd.DataFrame, filters: Any, group_by: str, column: str) -> pd.DataFrame:
    f_df = apply_filters(df, filters)
    return group_and_sum(f_df, group_by=group_by, column=column)


def filtered_group_and_average(df: pd.DataFrame, filters: Any, group_by: str, column: str) -> pd.DataFrame:
    f_df = apply_filters(df, filters)
    return group_and_average(f_df, group_by=group_by, column=column)


def filtered_value_counts(df: pd.DataFrame, filters: Any, column: str) -> pd.DataFrame:
    f_df = apply_filters(df, filters)
    return value_counts(f_df, column)


def top_n(df: pd.DataFrame, group_by: str, column: str, n: int = 5) -> List[Any]:
    df = validate_dataframe(df)
    res = group_and_sum(df, group_by, column)
    return res.sort_values(by=column, ascending=False).head(n)[column].tolist()


def filtered_top_n(df: pd.DataFrame, filters: Any, group_by: str, column: str, n: int = 5) -> List[Any]:
    f_df = apply_filters(df, filters)
    return top_n(f_df, group_by=group_by, column=column, n=n)


def percentage_of_total(df: pd.DataFrame, group_by: str, column: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    res = group_and_sum(df, group_by, column)
    total = res[column].sum()
    res["Percentage"] = (res[column] / total) * 100 if total != 0 else 0
    return res


def filtered_percentage_of_total(df: pd.DataFrame, filters: Any, group_by: str, column: str) -> pd.DataFrame:
    f_df = apply_filters(df, filters)
    return percentage_of_total(f_df, group_by=group_by, column=column)


def monthly_sum(df: pd.DataFrame, date_column: str, column: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column])
    res = temp.groupby(temp[date_column].dt.to_period("M"))[column].sum().reset_index()
    res[date_column] = res[date_column].astype(str)
    return res


def filtered_monthly_sum(df: pd.DataFrame, filters: Any, date_column: str, column: str) -> pd.DataFrame:
    f_df = apply_filters(df, filters)
    return monthly_sum(f_df, date_column=date_column, column=column)


def monthly_average(df: pd.DataFrame, date_column: str, column: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column])
    res = temp.groupby(temp[date_column].dt.to_period("M"))[column].mean().reset_index()
    res[date_column] = res[date_column].astype(str)
    return res


def filtered_monthly_average(df: pd.DataFrame, filters: Any, date_column: str, column: str) -> pd.DataFrame:
    f_df = apply_filters(df, filters)
    return monthly_average(f_df, date_column=date_column, column=column)


def monthly_count(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    df = validate_dataframe(df)
    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column])
    res = temp.groupby(temp[date_column].dt.to_period("M")).size().reset_index(name="Count")
    res[date_column] = res[date_column].astype(str)
    return res


def filtered_monthly_count(df: pd.DataFrame, filters: Any, date_column: str) -> pd.DataFrame:
    f_df = apply_filters(df, filters)
    return monthly_count(f_df, date_column=date_column)