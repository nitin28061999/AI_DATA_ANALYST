from __future__ import annotations

import contextlib
import numbers
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# GENERAL VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that df is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")


def validate_column(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Validate that a column exists."""
    validate_dataframe(df)

    if not isinstance(column, str) or not column.strip():
        raise ValueError("Column name cannot be empty.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )


def validate_numeric_column(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Validate that a column exists and is numeric."""
    validate_column(df, column)

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )


# ============================================================
# FILTER VALIDATION
# ============================================================

SUPPORTED_OPERATORS = {
    "=",
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "contains",
    "between",
}


def validate_filters(
    df: pd.DataFrame,
    filters: list,
) -> None:
    """
    Validate a list of filters.

    Filters use AND logic.
    Missing operator defaults to "=".
    """
    validate_dataframe(df)

    if not isinstance(filters, list):
        raise ValueError("Filters must be a list.")

    if not filters:
        raise ValueError("At least one filter is required.")

    for index, filter_item in enumerate(filters):

        if not isinstance(filter_item, dict):
            raise ValueError(
                f"Filter #{index + 1} must be a dictionary."
            )

        if "column" not in filter_item:
            raise ValueError(
                f"Filter #{index + 1} is missing 'column'."
            )

        if "value" not in filter_item:
            raise ValueError(
                f"Filter #{index + 1} is missing 'value'."
            )

        column = filter_item["column"]

        validate_column(
            df,
            column,
        )

        operator = filter_item.get(
            "operator",
            "=",
        )

        operator = _normalize_operator(operator)

        if operator not in {
            _normalize_operator(item)
            for item in SUPPORTED_OPERATORS
        }:
            raise ValueError(
                f"Unsupported filter operator '{operator}'. "
                f"Supported operators: "
                f"{sorted(SUPPORTED_OPERATORS)}"
            )

        if operator == "between":
            value = filter_item["value"]

            if not isinstance(
                value,
                (list, tuple),
            ) or len(value) != 2:
                raise ValueError(
                    "The 'between' operator requires "
                    "exactly two values."
                )


# ============================================================
# VALUE CONVERSION
# ============================================================

def _normalize_operator(operator: Any) -> str:
    """Normalize Gemini/user filter operators to supported operators."""

    if operator is None:
        return "="

    normalized = str(operator).strip().lower()

    # Gemini can occasionally repeat the equality character.
    # Treat any operator consisting only of "=" characters as "=".
    if normalized and set(normalized) == {"="}:
        return "="

    aliases = {
        "==": "=",
        "eq": "=",
        "equals": "=",
        "equal": "=",
        "not equal": "!=",
        "not_equal": "!=",
        "ne": "!=",
        "greater than": ">",
        "greater_than": ">",
        "gt": ">",
        "greater than or equal": ">=",
        "greater_than_or_equal": ">=",
        "gte": ">=",
        "less than": "<",
        "less_than": "<",
        "lt": "<",
        "less than or equal": "<=",
        "less_than_or_equal": "<=",
        "lte": "<=",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def _convert_value_for_series(
    series: pd.Series,
    value: Any,
) -> Any:
    """Convert a filter value to a type compatible with the pandas Series."""
    if pd.api.types.is_numeric_dtype(series):
        try:
            return pd.to_numeric(value)
        except (ValueError, TypeError):
            return value

    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            return pd.to_datetime(value)
        except (ValueError, TypeError):
            return value

    return value


def _is_text_series(
    series: pd.Series,
) -> bool:
    """Return True when a Series should be treated as text."""
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )


def _text_normalize(
    value: Any,
) -> str:
    """Normalize a value for case-insensitive text comparison."""
    return "" if value is None else str(value).strip().casefold()


# ============================================================
# FILTER ENGINE
# ============================================================

def apply_filters(
    df: pd.DataFrame,
    filters: list,
) -> pd.DataFrame:  # sourcery skip: low-code-quality
    """
    Apply multiple filters using AND logic.

    Supported operators: =, ==, !=, >, >=, <, <=, contains, between
    """
    validate_filters(
        df,
        filters,
    )

    mask = pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )

    for filter_item in filters:
        column = filter_item["column"]
        operator = _normalize_operator(
            filter_item.get("operator", "=")
        )
        value = filter_item["value"]
        series = df[column]

        # ====================================================
        # EQUAL
        # ====================================================
        if operator == "=":
            converted_value = _convert_value_for_series(series, value)

            if _is_text_series(series):
                current_mask = (
                    series.astype("string")
                    .str.strip()
                    .str.casefold()
                    == _text_normalize(converted_value)
                )
                current_mask = current_mask & series.notna()
            else:
                try:
                    current_mask = series == converted_value
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot compare column '{column}' with value '{value}'."
                    ) from exc

        # ====================================================
        # NOT EQUAL
        # ====================================================
        elif operator == "!=":
            converted_value = _convert_value_for_series(series, value)

            if _is_text_series(series):
                current_mask = (
                    series.astype("string")
                    .str.strip()
                    .str.casefold()
                    != _text_normalize(converted_value)
                )
            else:
                try:
                    current_mask = series != converted_value
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot compare column '{column}' with value '{value}'."
                    ) from exc

            current_mask = current_mask & series.notna()

        # ====================================================
        # GREATER THAN
        # ====================================================
        elif operator == ">":
            converted_value = _convert_value_for_series(series, value)
            try:
                current_mask = series > converted_value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot apply '>' to column '{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # GREATER THAN OR EQUAL
        # ====================================================
        elif operator == ">=":
            converted_value = _convert_value_for_series(series, value)
            try:
                current_mask = series >= converted_value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot apply '>=' to column '{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # LESS THAN
        # ====================================================
        elif operator == "<":
            converted_value = _convert_value_for_series(series, value)
            try:
                current_mask = series < converted_value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot apply '<' to column '{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # LESS THAN OR EQUAL
        # ====================================================
        elif operator == "<=":
            converted_value = _convert_value_for_series(series, value)
            try:
                current_mask = series <= converted_value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot apply '<=' to column '{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # CONTAINS
        # ====================================================
        elif operator == "contains":
            current_mask = (
                series.astype("string")
                .str.contains(
                    str(value),
                    case=False,
                    na=False,
                    regex=False,
                )
            )

        # ====================================================
        # BETWEEN
        # ====================================================
        elif operator == "between":
            lower_value = _convert_value_for_series(series, value[0])
            upper_value = _convert_value_for_series(series, value[1])

            try:
                current_mask = series.between(
                    lower_value,
                    upper_value,
                    inclusive="both",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot apply 'between' to column '{column}' "
                    f"with values '{value[0]}' and '{value[1]}'."
                ) from exc

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        mask = mask & current_mask.fillna(False).astype(bool)

    return df.loc[mask].copy()


# ============================================================
# BASIC AGGREGATIONS
# ============================================================

def calculate_sum(
    df: pd.DataFrame,
    column: str,
) -> float:
    """Calculate the sum of a numeric column."""
    validate_numeric_column(df, column)
    return float(df[column].sum())


def calculate_average(
    df: pd.DataFrame,
    column: str,
) -> float:
    """Calculate the average of a numeric column."""
    validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column '{column}' contains no numeric values.")

    return float(values.mean())


def calculate_count(
    df: pd.DataFrame,
    column: str,
) -> int:
    """Count non-null values in a column."""
    validate_column(df, column)
    return int(df[column].count())


def calculate_unique_count(
    df: pd.DataFrame,
    column: str,
) -> int:
    """Count unique non-null values."""
    validate_column(df, column)
    return int(df[column].nunique(dropna=True))


def calculate_min(
    df: pd.DataFrame,
    column: str,
) -> Any:
    """Return the minimum non-null value."""
    validate_column(df, column)
    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column '{column}' contains no values.")

    return values.min()


def calculate_max(
    df: pd.DataFrame,
    column: str,
) -> Any:
    """Return the maximum non-null value."""
    validate_column(df, column)
    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column '{column}' contains no values.")

    return values.max()


# ============================================================
# GROUP ANALYSIS
# ============================================================

def group_and_sum(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Group by a column and calculate sums.
    Special handling is included when group_column and value_column are identical.
    """
    validate_column(df, group_column)
    validate_numeric_column(df, value_column)

    if group_column == value_column:
        result = (
            df.groupby(group_column, dropna=False)
            .agg(**{"Sum": (value_column, "sum")})
            .reset_index()
        )
        return (
            result.sort_values(by="Sum", ascending=False)
            .reset_index(drop=True)
        )

    result = (
        df.groupby(group_column, dropna=False, as_index=False)[value_column]
        .sum()
        .sort_values(by=value_column, ascending=False) # pyright: ignore[reportCallIssue]
        .reset_index(drop=True)
    )

    return result


def group_and_average(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Group by a column and calculate averages.
    Handles the case where group_column and value_column are identical.
    """
    validate_column(df, group_column)
    validate_numeric_column(df, value_column)

    if group_column == value_column:
        result = (
            df.groupby(group_column, dropna=False)
            .agg(**{"Average": (value_column, "mean")})
            .reset_index()
        )
        return (
            result.sort_values(by="Average", ascending=False)
            .reset_index(drop=True)
        )

    result = (
        df.groupby(group_column, dropna=False, as_index=False)[value_column]
        .mean()
        .sort_values(by=value_column, ascending=False) # pyright: ignore[reportCallIssue]
        .reset_index(drop=True)
    )

    return result


def group_and_count(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Group by a column and count rows."""
    validate_column(df, group_column)

    result = (
        df.groupby(group_column, dropna=False)
        .size()
        .reset_index(name="Count")
    )

    return (
        result.sort_values(by="Count", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# TOP N
# ============================================================

def top_n(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top N groups ranked by summed value."""
    validate_column(df, group_column)
    validate_numeric_column(df, value_column)

    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise ValueError("n must be an integer.")

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    result = group_and_sum(df, group_column, value_column)
    return result.head(n).reset_index(drop=True)


# ============================================================
# PERCENTAGE OF TOTAL
# ============================================================

def _calculate_percentage_and_sort(
    grouped_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Internal helper to calculate percentages and sort descending."""
    total = grouped_df[value_col].sum()
    grouped_df["Percentage"] = (
        0.0 if total == 0 else (grouped_df[value_col] / total) * 100
    )
    return (
        grouped_df.sort_values(by="Percentage", ascending=False)
        .reset_index(drop=True)
    )


def percentage_of_total(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Calculate each group's percentage of the total."""
    validate_column(df, group_column)
    validate_numeric_column(df, value_column)

    if group_column == value_column:
        grouped = (
            df.groupby(group_column, dropna=False)[value_column]
            .sum()
            .rename("Value")
            .reset_index()
        )
        return _calculate_percentage_and_sort(grouped, "Value")

    grouped = (
        df.groupby(group_column, dropna=False, as_index=False)[value_column]
        .sum()
    )
    return _calculate_percentage_and_sort(grouped, value_column) # pyright: ignore[reportArgumentType]


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

def monthly_sum(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Calculate monthly sums for a numeric value column."""
    validate_column(df, date_column)
    validate_numeric_column(df, value_column)

    working_df = df.copy()
    working_df[date_column] = pd.to_datetime(
        working_df[date_column],
        errors="coerce",
    )
    working_df = working_df.dropna(subset=[date_column])

    if working_df.empty:
        raise ValueError(
            f"Column '{date_column}' contains no valid dates."
        )

    working_df["Month"] = (
        working_df[date_column]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        working_df.groupby("Month", as_index=False)[value_column]
        .sum()
    )

    return result.sort_values("Month").reset_index(drop=True) # pyright: ignore[reportCallIssue]


# ============================================================
# VALUE COUNTS
# ============================================================

def value_counts(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return counts for each distinct value."""
    validate_column(df, column)

    result = (
        df[column]
        .value_counts(dropna=False)
        .rename("Count")
        .reset_index()
    )
    result.columns = [column, "Count"]
    return result


# ============================================================
# FILTERED AGGREGATIONS
# ============================================================

def filtered_sum(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> float:
    """Calculate a sum after applying filters."""
    validate_numeric_column(df, value_column)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    return float(filtered_df[value_column].sum())


def filtered_average(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> float:
    """Calculate an average after applying filters."""
    validate_numeric_column(df, value_column)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    values = filtered_df[value_column].dropna()

    if values.empty:
        raise ValueError(
            f"No numeric values remain in '{value_column}' after filtering."
        )

    return float(values.mean())


def filtered_count(
    df: pd.DataFrame,
    filters: list,
    count_column: str,
) -> int:
    """Count non-null values after applying filters."""
    validate_column(df, count_column)
    filtered_df = apply_filters(df, filters)

    return 0 if filtered_df.empty else int(filtered_df[count_column].count())


def filtered_unique_count(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> int:
    """Count unique values after applying filters."""
    validate_column(df, value_column)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        return 0

    return int(filtered_df[value_column].nunique(dropna=True))


def filtered_min(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> Any:
    """Return the minimum value after filtering."""
    validate_column(df, value_column)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    values = filtered_df[value_column].dropna()

    if values.empty:
        raise ValueError(
            f"No values remain in '{value_column}' after filtering."
        )

    return values.min()


def filtered_max(
    df: pd.DataFrame,
    filters: list,
    value_column: str,
) -> Any:
    """Return the maximum value after filtering."""
    validate_column(df, value_column)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    values = filtered_df[value_column].dropna()

    if values.empty:
        raise ValueError(
            f"No values remain in '{value_column}' after filtering."
        )

    return values.max()


# ============================================================
# FILTERED GROUP OPERATIONS
# ============================================================

def filtered_group_and_sum(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Group and sum after applying filters."""
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    return group_and_sum(
        filtered_df,
        group_column,
        value_column,
    )


def filtered_group_and_average(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Group and average after applying filters."""
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    return group_and_average(
        filtered_df,
        group_column,
        value_column,
    )


def filtered_value_counts(
    df: pd.DataFrame,
    filters: list,
    column: str,
) -> pd.DataFrame:
    """Return value counts after applying filters."""
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    return value_counts(filtered_df, column)


def filtered_top_n(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
    n: int = 10,
) -> pd.DataFrame:
    """Return top N groups after applying filters."""
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        raise ValueError("No rows matched the specified filters.")

    return top_n(
        filtered_df,
        group_column,
        value_column,
        n,
    )


# ============================================================
# DATAFRAME SERIALIZATION HELPER
# ============================================================

def dataframe_to_records(result: Any) -> Any:
    """Convert pandas/numpy results into JSON-safe Python objects."""
    if isinstance(result, pd.DataFrame):
        return [
            dataframe_to_records(row)
            for row in result.to_dict(orient="records")
        ]

    if isinstance(result, pd.Series):
        return {
            str(key): dataframe_to_records(val)
            for key, val in result.to_dict().items()
        }

    if isinstance(result, pd.Timestamp):
        return result.isoformat()

    if isinstance(result, pd.Timedelta):
        return result.total_seconds()

    if isinstance(result, np.generic):
        return dataframe_to_records(result.item())

    if isinstance(result, np.ndarray):
        return [dataframe_to_records(item) for item in result.tolist()]

    if isinstance(result, dict):
        return {
            str(key): dataframe_to_records(val)
            for key, val in result.items()
        }

    if isinstance(result, (list, tuple)):
        return [dataframe_to_records(item) for item in result]

    if result is None:
        return None

    with contextlib.suppress(TypeError, ValueError):
        missing = pd.isna(result)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None

    return result