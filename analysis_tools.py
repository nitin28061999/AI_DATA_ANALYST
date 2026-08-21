import pandas as pd


# ============================================================
# HELPER
# ============================================================

def validate_column(df, column):
    """Check whether a column exists in the dataframe."""

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )


def numeric_column(df, column):
    """Convert a column to numeric safely."""

    validate_column(df, column)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# BASIC OPERATIONS
# ============================================================

def calculate_sum(df, column):

    values = numeric_column(df, column)

    return float(values.sum())


def calculate_average(df, column):

    values = numeric_column(df, column)

    return float(values.mean())


def calculate_count(df, column=None):

    if column is None:
        return len(df)

    validate_column(df, column)

    return int(df[column].count())


def calculate_min(df, column):

    values = numeric_column(df, column)

    return float(values.min())


def calculate_max(df, column):

    values = numeric_column(df, column)

    return float(values.max())


# ============================================================
# GROUP AND SUM
# ============================================================

def group_and_sum(
    df,
    group_column,
    value_column
):

    validate_column(df, group_column)
    validate_column(df, value_column)

    temp_df = df[
        [group_column, value_column]
    ].copy()

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    return (
        temp_df.groupby(group_column, dropna=False)[value_column]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


# ============================================================
# TOP N
# ============================================================

def top_n(
    df,
    group_column,
    value_column,
    n=10
):

    validate_column(df, group_column)
    validate_column(df, value_column)

    n = int(n)

    if n <= 0:
        raise ValueError(
            "n must be greater than 0."
        )

    temp_df = df[
        [group_column, value_column]
    ].copy()

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    return (
        temp_df.groupby(group_column, dropna=False)[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )


# ============================================================
# GROUP AND AVERAGE
# ============================================================

def group_and_average(
    df,
    group_column,
    value_column
):

    validate_column(df, group_column)
    validate_column(df, value_column)

    temp_df = df[
        [group_column, value_column]
    ].copy()

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    return (
        temp_df.groupby(group_column, dropna=False)[value_column]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )


# ============================================================
# PERCENTAGE OF TOTAL
# ============================================================

def percentage_of_total(
    df,
    group_column,
    value_column
):

    validate_column(df, group_column)
    validate_column(df, value_column)

    temp_df = df[
        [group_column, value_column]
    ].copy()

    temp_df[value_column] = pd.to_numeric(
        temp_df[value_column],
        errors="coerce"
    )

    grouped = (
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

    total = grouped[value_column].sum()

    if total == 0:
        grouped["Percentage"] = 0.0
    else:
        grouped["Percentage"] = (
            grouped[value_column] / total * 100
        )

    return grouped


# ============================================================
# FILTER AND SUM
# ============================================================

def filter_and_sum(
    df,
    filter_column,
    filter_value,
    value_column
):

    validate_column(df, filter_column)
    validate_column(df, value_column)

    filtered = df[
        df[filter_column]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        str(filter_value)
        .lower()
        .strip()
    ]

    values = pd.to_numeric(
        filtered[value_column],
        errors="coerce"
    )

    return float(values.sum())


# ============================================================
# MONTHLY SUM
# ============================================================

def monthly_sum(
    df,
    date_column,
    value_column
):

    validate_column(df, date_column)
    validate_column(df, value_column)

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
            date_column,
            value_column
        ]
    )

    temp_df["Month"] = (
        temp_df[date_column]
        .dt.to_period("M")
        .astype(str)
    )

    return temp_df.groupby("Month")[value_column].sum().reset_index()