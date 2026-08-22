# analysis_tools.py

from __future__ import annotations

import pandas as pd
import numpy as np


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
    column: str
) -> None:
    """Validate that a column exists."""

    validate_dataframe(df)

    if column is None or not column.strip():
        raise ValueError("Column name cannot be empty.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )


def validate_numeric_column(
    df: pd.DataFrame,
    column: str
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
    filters: list
) -> None:
    """
    Validate filters.

    IMPORTANT:
    Missing operator is intentionally allowed and treated as "=".

    This makes the analysis engine compatible with plans such as:

        {
            "column": "City",
            "value": "Delhi"
        }

    and also:

        {
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }
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
            column
        )

        # ----------------------------------------------------
        # Missing operator means equality.
        # ----------------------------------------------------

        operator = filter_item.get(
            "operator",
            "="
        )

        if operator is None:
            operator = "="

        if not isinstance(operator, str):
            raise ValueError(
                f"Filter #{index + 1} operator must be a string."
            )

        operator = operator.strip().lower()

        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Unsupported filter operator '{operator}'. "
                f"Supported operators: "
                f"{sorted(SUPPORTED_OPERATORS)}"
            )

        # ----------------------------------------------------
        # Validate BETWEEN
        # ----------------------------------------------------

        if operator == "between":

            value = filter_item["value"]

            if not isinstance(
                value,
                (list, tuple)
            ):
                raise ValueError(
                    "The 'between' operator requires "
                    "exactly two values."
                )

            if len(value) != 2:
                raise ValueError(
                    "The 'between' operator requires "
                    "exactly two values."
                )


# ============================================================
# VALUE CONVERSION
# ============================================================

def _normalize_operator(
    operator
) -> str:
    """Normalize operator."""

    if operator is None:
        return "="

    operator = str(
        operator
    ).strip().lower()

    return "=" if operator == "==" else operator


def _convert_value_for_series(
    series: pd.Series,
    value
):
    """Convert filter values to compatible pandas types."""

    if pd.api.types.is_numeric_dtype(series):

        try:
            return pd.to_numeric(value)
        except (
            ValueError,
            TypeError
        ):
            return value

    if pd.api.types.is_datetime64_any_dtype(series):

        try:
            return pd.to_datetime(value)
        except (
            ValueError,
            TypeError
        ):
            return value

    return value


# ============================================================
# FILTER ENGINE
# ============================================================

def apply_filters(
    df: pd.DataFrame,
    filters: list
) -> pd.DataFrame:  # sourcery skip: low-code-quality
    """
    Apply multiple filters using AND logic.

    Missing operator defaults to "=".

    Examples:

        [
            {
                "column": "City",
                "value": "Delhi"
            }
        ]

    or:

        [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi"
            },
            {
                "column": "Revenue",
                "operator": ">",
                "value": 1000
            }
        ]
    """

    validate_filters(
        df,
        filters
    )

    mask = pd.Series(
        True,
        index=df.index
    )

    for filter_item in filters:

        column = filter_item["column"]

        operator = _normalize_operator(
            filter_item.get(
                "operator",
                "="
            )
        )

        value = filter_item["value"]

        series = df[column]

        # ====================================================
        # EQUAL
        # ====================================================

        if operator == "=":

            converted_value = (
                _convert_value_for_series(
                    series,
                    value
                )
            )

            # Case-insensitive equality for text.
            if (
                pd.api.types.is_string_dtype(series)
                or series.dtype == object
            ):
                current_mask = (
                    series.astype(str)
                    .str.strip()
                    .str.casefold()
                    ==
                    str(converted_value)
                    .strip()
                    .casefold()
                )

                # Preserve NaN behavior.
                current_mask = current_mask & series.notna()

            else:
                current_mask = (
                    series == converted_value
                )

        # ====================================================
        # NOT EQUAL
        # ====================================================

        elif operator == "!=":

            converted_value = (
                _convert_value_for_series(
                    series,
                    value
                )
            )

            if (
                pd.api.types.is_string_dtype(series)
                or series.dtype == object
            ):
                current_mask = (
                    series.astype(str)
                    .str.strip()
                    .str.casefold()
                    !=
                    str(converted_value)
                    .strip()
                    .casefold()
                )

                current_mask = current_mask & series.notna()

            else:
                current_mask = (
                    series != converted_value
                )

        # ====================================================
        # GREATER THAN
        # ====================================================

        elif operator == ">":

            converted_value = (
                _convert_value_for_series(
                    series,
                    value
                )
            )

            try:
                current_mask = (
                    series > converted_value
                )
            except TypeError as exc:
                raise ValueError(
                    f"Cannot apply '>' to column "
                    f"'{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # GREATER THAN OR EQUAL
        # ====================================================

        elif operator == ">=":

            converted_value = (
                _convert_value_for_series(
                    series,
                    value
                )
            )

            try:
                current_mask = (
                    series >= converted_value
                )
            except TypeError as exc:
                raise ValueError(
                    f"Cannot apply '>=' to column "
                    f"'{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # LESS THAN
        # ====================================================

        elif operator == "<":

            converted_value = (
                _convert_value_for_series(
                    series,
                    value
                )
            )

            try:
                current_mask = (
                    series < converted_value
                )
            except TypeError as exc:
                raise ValueError(
                    f"Cannot apply '<' to column "
                    f"'{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # LESS THAN OR EQUAL
        # ====================================================

        elif operator == "<=":

            converted_value = (
                _convert_value_for_series(
                    series,
                    value
                )
            )

            try:
                current_mask = (
                    series <= converted_value
                )
            except TypeError as exc:
                raise ValueError(
                    f"Cannot apply '<=' to column "
                    f"'{column}' with value '{value}'."
                ) from exc

        # ====================================================
        # CONTAINS
        # ====================================================

        elif operator == "contains":

            current_mask = (
                series.astype(str)
                .str.contains(
                    str(value),
                    case=False,
                    na=False,
                    regex=False
                )
            )

        # ====================================================
        # BETWEEN
        # ====================================================

        elif operator == "between":

            lower_value = _convert_value_for_series(
                series,
                value[0]
            )

            upper_value = _convert_value_for_series(
                series,
                value[1]
            )

            try:

                current_mask = series.between(
                    lower_value,
                    upper_value,
                    inclusive="both"
                )

            except TypeError as exc:

                raise ValueError(
                    f"Cannot apply 'between' to "
                    f"column '{column}'."
                ) from exc

        else:

            raise ValueError(
                f"Unsupported operator: {operator}"
            )

        mask = (
            mask
            & current_mask.fillna(False)
        )

    return df.loc[mask].copy()


# ============================================================
# BASIC AGGREGATIONS
# ============================================================

def calculate_sum(
    df: pd.DataFrame,
    column: str
) -> float:

    validate_numeric_column(
        df,
        column
    )

    return float(
        df[column].sum()
    )


def calculate_average(
    df: pd.DataFrame,
    column: str
) -> float:

    validate_numeric_column(
        df,
        column
    )

    if df[column].dropna().empty:
        raise ValueError(
            f"Column '{column}' contains no numeric values."
        )

    return float(
        df[column].mean()
    )


def calculate_count(
    df: pd.DataFrame,
    column: str
) -> int:

    validate_column(
        df,
        column
    )

    return int(
        df[column].count()
    )


def calculate_unique_count(
    df: pd.DataFrame,
    column: str
) -> int:

    validate_column(
        df,
        column
    )

    return int(
        df[column].nunique(
            dropna=True
        )
    )


def calculate_min(
    df: pd.DataFrame,
    column: str
):

    validate_column(
        df,
        column
    )

    if df[column].dropna().empty:
        raise ValueError(
            f"Column '{column}' contains no values."
        )

    return df[column].min()


def calculate_max(
    df: pd.DataFrame,
    column: str
):

    validate_column(
        df,
        column
    )

    if df[column].dropna().empty:
        raise ValueError(
            f"Column '{column}' contains no values."
        )

    return df[column].max()


# ============================================================
# GROUP ANALYSIS
# ============================================================

def group_and_sum(
    df: pd.DataFrame,
    group_column: str,
    value_column: str
) -> pd.DataFrame:

    validate_column(
        df,
        group_column
    )

    validate_numeric_column(
        df,
        value_column
    )

    result = (
        df.groupby(
            group_column,
            dropna=False
        )[value_column]
        .sum()
        .reset_index()
    )

    return (
        result
        .sort_values(
            by=value_column,
            ascending=False
        )
        .reset_index(drop=True)
    )


def group_and_average(
    df: pd.DataFrame,
    group_column: str,
    value_column: str
) -> pd.DataFrame:

    validate_column(
        df,
        group_column
    )

    validate_numeric_column(
        df,
        value_column
    )

    result = (
        df.groupby(
            group_column,
            dropna=False
        )[value_column]
        .mean()
        .reset_index()
    )

    return (
        result
        .sort_values(
            by=value_column,
            ascending=False
        )
        .reset_index(drop=True)
    )


def group_and_count(
    df: pd.DataFrame,
    group_column: str
) -> pd.DataFrame:

    validate_column(
        df,
        group_column
    )

    result = (
        df.groupby(
            group_column,
            dropna=False
        )
        .size()
        .reset_index(
            name="Count"
        )
    )

    return (
        result
        .sort_values(
            by="Count",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# TOP N
# ============================================================

def top_n(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    n: int = 10
) -> pd.DataFrame:

    validate_column(
        df,
        group_column
    )

    validate_numeric_column(
        df,
        value_column
    )

    try:
        n = n
    except (
        ValueError,
        TypeError
    ) as exc:
        raise ValueError(
            "n must be an integer."
        ) from exc

    if n <= 0:
        raise ValueError(
            "n must be greater than zero."
        )

    result = group_and_sum(
        df,
        group_column,
        value_column
    )

    return (
        result
        .head(n)
        .reset_index(drop=True)
    )


# ============================================================
# PERCENTAGE OF TOTAL
# ============================================================

def percentage_of_total(
    df: pd.DataFrame,
    group_column: str,
    value_column: str
) -> pd.DataFrame:

    validate_column(
        df,
        group_column
    )

    validate_numeric_column(
        df,
        value_column
    )

    grouped = (
        df.groupby(
            group_column,
            dropna=False
        )[value_column]
        .sum()
        .reset_index()
    )

    total = grouped[value_column].sum()

    if total == 0:

        grouped["Percentage"] = 0.0

    else:

        grouped["Percentage"] = (
            grouped[value_column]
            / total
            * 100
        )

    return (
        grouped
        .sort_values(
            by="Percentage",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

def monthly_sum(
    df: pd.DataFrame,
    date_column: str,
    value_column: str
) -> pd.DataFrame:

    validate_column(
        df,
        date_column
    )

    validate_numeric_column(
        df,
        value_column
    )

    working_df = df.copy()

    working_df[date_column] = pd.to_datetime(
        working_df[date_column],
        errors="coerce"
    )

    working_df = working_df.dropna(
        subset=[date_column]
    )

    if working_df.empty:
        raise ValueError(
            f"Column '{date_column}' contains "
            "no valid dates."
        )

    working_df["Month"] = (
        working_df[date_column]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        working_df
        .groupby("Month")[value_column]
        .sum()
        .reset_index()
    )

    return (
        result
        .sort_values("Month")
        .reset_index(drop=True)
    )


# ============================================================
# VALUE COUNTS
# ============================================================

def value_counts(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:

    validate_column(
        df,
        column
    )

    result = (
        df[column]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    result.columns = [
        column,
        "Count"
    ]

    return result


# ============================================================
# FILTERED AGGREGATIONS
# ============================================================

def filtered_sum(
    df: pd.DataFrame,
    filters: list,
    value_column: str
) -> float:

    validate_numeric_column(
        df,
        value_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    return float(
        filtered_df[value_column].sum()
    )


def filtered_average(
    df: pd.DataFrame,
    filters: list,
    value_column: str
) -> float:

    validate_numeric_column(
        df,
        value_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    values = filtered_df[
        value_column
    ].dropna()

    if values.empty:
        raise ValueError(
            f"No numeric values remain in "
            f"'{value_column}' after filtering."
        )

    return float(
        values.mean()
    )


def filtered_count(
    df: pd.DataFrame,
    filters: list,
    count_column: str
) -> int:

    validate_column(
        df,
        count_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    return int(
        filtered_df[count_column].count()
    )


def filtered_unique_count(
    df: pd.DataFrame,
    filters: list,
    value_column: str
) -> int:

    validate_column(
        df,
        value_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    return int(
        filtered_df[value_column]
        .nunique(dropna=True)
    )


def filtered_min(
    df: pd.DataFrame,
    filters: list,
    value_column: str
):

    validate_column(
        df,
        value_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    values = filtered_df[
        value_column
    ].dropna()

    if values.empty:
        raise ValueError(
            f"No values remain in "
            f"'{value_column}' after filtering."
        )

    return values.min()


def filtered_max(
    df: pd.DataFrame,
    filters: list,
    value_column: str
):

    validate_column(
        df,
        value_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    values = filtered_df[
        value_column
    ].dropna()

    if values.empty:
        raise ValueError(
            f"No values remain in "
            f"'{value_column}' after filtering."
        )

    return values.max()


# ============================================================
# FILTERED GROUP OPERATIONS
# ============================================================

def filtered_group_and_sum(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str
) -> pd.DataFrame:

    filtered_df = apply_filters(
        df,
        filters
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return group_and_sum(
        filtered_df,
        group_column,
        value_column
    )


def filtered_group_and_average(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str
) -> pd.DataFrame:

    filtered_df = apply_filters(
        df,
        filters
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return group_and_average(
        filtered_df,
        group_column,
        value_column
    )


def filtered_value_counts(
    df: pd.DataFrame,
    filters: list,
    column: str
) -> pd.DataFrame:

    filtered_df = apply_filters(
        df,
        filters
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return value_counts(
        filtered_df,
        column
    )


def filtered_top_n(
    df: pd.DataFrame,
    filters: list,
    group_column: str,
    value_column: str,
    n: int = 10
) -> pd.DataFrame:

    filtered_df = apply_filters(
        df,
        filters
    )

    if filtered_df.empty:
        raise ValueError(
            "No rows matched the specified filters."
        )

    return top_n(
        filtered_df,
        group_column,
        value_column,
        n
    )


# ============================================================
# DATAFRAME SERIALIZATION HELPER
# ============================================================

def dataframe_to_records(
    result
):
    """
    Convert pandas results into JSON-safe Python objects.
    """

    if isinstance(result, pd.DataFrame):

        return result.to_dict(
            orient="records"
        )

    if isinstance(result, pd.Series):

        return result.to_dict()

    if isinstance(result, np.generic):

        return result.item()

    if isinstance(result, dict):

        return {
            str(key): dataframe_to_records(value)
            for key, value in result.items()
        }

    if isinstance(result, list):

        return [
            dataframe_to_records(item)
            for item in result
        ]

    return result