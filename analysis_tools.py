import pandas as pd


def calculate_sum(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.sum())


def calculate_average(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.mean())


def calculate_count(df, column=None):
    if column is None:
        return len(df)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    return int(df[column].count())


def calculate_min(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.min())


def calculate_max(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(values.max())


def group_and_sum(
    df,
    group_column,
    value_column
):
    if group_column not in df.columns:
        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
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
        .reset_index()
    )


def top_n(
    df,
    group_column,
    value_column,
    n=10
):
    """
    Return the top N groups based on
    the sum of a numeric value column.
    """

    if group_column not in df.columns:
        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
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