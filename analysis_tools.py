import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def validate_column(df, column):
    """Validate that a column exists in the dataframe."""

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )


def validate_filters(df, filters):
    """Validate multi-filter definitions."""

    if not isinstance(filters, list):
        raise ValueError(
            "Filters must be provided as a list."
        )

    if not filters:
        raise ValueError(
            "At least one filter is required."
        )

    for filter_item in filters:

        if not isinstance(filter_item, dict):
            raise ValueError(
                "Each filter must be a dictionary."
            )

        column = filter_item.get("column")

        if not column:
            raise ValueError(
                "Each filter must contain a column."
            )

        validate_column(
            df,
            column
        )


# ============================================================
# BASIC OPERATIONS
# ============================================================

def calculate_sum(df, column):

    validate_column(df, column)

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.sum())


def calculate_average(df, column):

    validate_column(df, column)

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.mean())


def calculate_count(df, column=None):

    if column is None:
        return len(df)

    validate_column(
        df,
        column
    )

    return int(
        df[column].count()
    )


def calculate_unique_count(df, column):

    validate_column(
        df,
        column
    )

    return int(
        df[column].nunique(
            dropna=True
        )
    )


def calculate_min(df, column):

    validate_column(
        df,
        column
    )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.min())


def calculate_max(df, column):

    validate_column(
        df,
        column
    )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.max())


# ============================================================
# GROUP OPERATIONS
# ============================================================

def group_and_sum(
    df,
    group_column,
    value_column
):

    validate_column(
        df,
        group_column
    )

    validate_column(
        df,
        value_column
    )

    temp_df = df[
        [group_column, value_column]
    ].copy()

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    return (
        temp_df
        .groupby(
            group_column,
            dropna=False
        )[value_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


def group_and_average(
    df,
    group_column,
    value_column
):

    validate_column(
        df,
        group_column
    )

    validate_column(
        df,
        value_column
    )

    temp_df = df[
        [group_column, value_column]
    ].copy()

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    return (
        temp_df
        .groupby(
            group_column,
            dropna=False
        )[value_column]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


def group_and_count(
    df,
    group_column
):

    validate_column(
        df,
        group_column
    )

    return (
        df[group_column]
        .value_counts(
            dropna=False
        )
        .rename("Count")
        .reset_index()
    )


def top_n(
    df,
    group_column,
    value_column,
    n=10
):

    validate_column(
        df,
        group_column
    )

    validate_column(
        df,
        value_column
    )

    temp_df = df[
        [group_column, value_column]
    ].copy()

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    return (
        temp_df
        .groupby(
            group_column,
            dropna=False
        )[value_column]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(int(n))
        .reset_index()
    )


# ============================================================
# PERCENTAGE
# ============================================================

def percentage_of_total(
    df,
    group_column,
    value_column
):

    result = group_and_sum(
        df,
        group_column,
        value_column
    )

    total = result[value_column].sum()

    if total == 0:
        result["Percentage"] = 0.0

    else:
        result["Percentage"] = (
            result[value_column]
            / total
            * 100
        )

    return result


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

def monthly_sum(
    df,
    date_column,
    value_column
):

    validate_column(
        df,
        date_column
    )

    validate_column(
        df,
        value_column
    )

    temp_df = df[
        [date_column, value_column]
    ].copy()

    temp_df[date_column] = pd.to_datetime(
        temp_df[date_column],
        errors="coerce"
    )

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    temp_df = temp_df.dropna(
        subset=[
            date_column
        ]
    )

    result = (
        temp_df
        .groupby(
            temp_df[date_column].dt.to_period("M")
        )[value_column]
        .sum()
        .reset_index()
    )

    result[date_column] = (
        result[date_column]
        .astype(str)
    )

    return result


# ============================================================
# VALUE COUNTS
# ============================================================

def value_counts(
    df,
    column
):

    validate_column(
        df,
        column
    )

    return (
        df[column]
        .value_counts(
            dropna=False
        )
        .rename("Count")
        .reset_index()
    )


# ============================================================
# MULTI-FILTER ENGINE
# ============================================================

def apply_filters(
    df,
    filters
):
    """
    Apply multiple AND filters.

    Example:

    [
        {"column": "City", "value": "Delhi"},
        {"column": "Product", "value": "Laptop"}
    ]
    """

    validate_filters(
        df,
        filters
    )

    filtered_df = df.copy()

    for filter_item in filters:

        column = filter_item["column"]
        value = filter_item["value"]

        filtered_df = filtered_df[
            filtered_df[column].astype(str).str.strip().str.lower()
            == str(value).strip().lower()
        ]

    return filtered_df


# ============================================================
# FILTERED SUM
# ============================================================

def filtered_sum(
    df,
    filters,
    value_column
):

    validate_column(
        df,
        value_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    values = pd.to_numeric(
        filtered_df[value_column],
        errors="coerce"
    )

    return float(
        values.sum()
    )


# ============================================================
# FILTERED AVERAGE
# ============================================================

def filtered_average(
    df,
    filters,
    value_column
):

    validate_column(
        df,
        value_column
    )

    filtered_df = apply_filters(
        df,
        filters
    )

    values = pd.to_numeric(
        filtered_df[value_column],
        errors="coerce"
    )

    return float(
        values.mean()
    )


# ============================================================
# FILTERED COUNT
# ============================================================

def filtered_count(
    df,
    filters,
    count_column=None
):

    filtered_df = apply_filters(
        df,
        filters
    )

    if count_column is None:
        return len(filtered_df)

    validate_column(
        filtered_df,
        count_column
    )

    return int(
        filtered_df[count_column].count()
    )


# ============================================================
# FILTERED UNIQUE COUNT
# ============================================================

def filtered_unique_count(
    df,
    filters,
    value_column
):

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
        .nunique(
            dropna=True
        )
    )