import pandas as pd


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_column(df, column):
    """
    Check whether a column exists in the dataset.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )


def numeric_series(df, column):
    """
    Convert a dataframe column to numeric values.
    Invalid values become NaN.
    """

    validate_column(
        df,
        column
    )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    if values.notna().sum() == 0:
        raise ValueError(
            f"Column '{column}' does not contain numeric values."
        )

    return values


# ============================================================
# SUM
# ============================================================

def calculate_sum(df, column):
    """
    Calculate the total of a numeric column.
    """

    values = numeric_series(
        df,
        column
    )

    return float(
        values.sum()
    )


# ============================================================
# AVERAGE
# ============================================================

def calculate_average(df, column):
    """
    Calculate the average of a numeric column.
    """

    values = numeric_series(
        df,
        column
    )

    return float(
        values.mean()
    )


# ============================================================
# COUNT
# ============================================================

def calculate_count(df, column=None):
    """
    Count non-null values in a column.

    If column is None, count total rows.
    """

    if column is None:
        return len(df)

    validate_column(
        df,
        column
    )

    return int(
        df[column].count()
    )


# ============================================================
# UNIQUE COUNT
# ============================================================

def calculate_unique_count(df, column):
    """
    Count the number of unique/non-duplicate values
    in a column.

    Example:

    Customer_ID:
    101
    101
    102
    103

    Result:
    3
    """

    validate_column(
        df,
        column
    )

    return int(
        df[column].nunique(
            dropna=True
        )
    )


# ============================================================
# MINIMUM
# ============================================================

def calculate_min(df, column):
    """
    Calculate the minimum value.
    """

    values = numeric_series(
        df,
        column
    )

    return float(
        values.min()
    )


# ============================================================
# MAXIMUM
# ============================================================

def calculate_max(df, column):
    """
    Calculate the maximum value.
    """

    values = numeric_series(
        df,
        column
    )

    return float(
        values.max()
    )


# ============================================================
# GROUP + SUM
# ============================================================

def group_and_sum(
    df,
    group_column,
    value_column
):
    """
    Group data by one column and calculate
    the sum of another column.
    """

    validate_column(
        df,
        group_column
    )

    values = numeric_series(
        df,
        value_column
    )

    temp_df = df[
        [group_column]
    ].copy()

    temp_df[value_column] = values

    return (
        temp_df.groupby(group_column, dropna=False)[value_column]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


# ============================================================
# GROUP + AVERAGE
# ============================================================

def group_and_average(
    df,
    group_column,
    value_column
):
    """
    Group data by one column and calculate
    the average of another column.
    """

    validate_column(
        df,
        group_column
    )

    values = numeric_series(
        df,
        value_column
    )

    temp_df = df[
        [group_column]
    ].copy()

    temp_df[value_column] = values

    return (
        temp_df.groupby(group_column, dropna=False)[value_column]
        .mean()
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
    """
    Return the top N groups based on
    the sum of a numeric column.
    """

    result = group_and_sum(
        df,
        group_column,
        value_column
    )

    return result.head(
        int(n)
    )


# ============================================================
# PERCENTAGE OF TOTAL
# ============================================================

def percentage_of_total(
    df,
    group_column,
    value_column
):
    """
    Calculate each group's percentage
    contribution to the total.
    """

    result = group_and_sum(
        df,
        group_column,
        value_column
    )

    total = result[
        value_column
    ].sum()

    if total == 0:

        result[
            "Percentage"
        ] = 0.0

    else:

        result[
            "Percentage"
        ] = (
            result[value_column]
            / total
            * 100
        )

    return result


# ============================================================
# MONTHLY SUM
# ============================================================

def monthly_sum(
    df,
    date_column,
    value_column
):
    """
    Calculate monthly totals.

    Uses 'ME' instead of the deprecated
    Pandas 'M' frequency.
    """

    validate_column(
        df,
        date_column
    )

    values = numeric_series(
        df,
        value_column
    )

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    temp_df = pd.DataFrame(
        {
            "Date": dates,
            value_column: values,
        }
    )

    temp_df = temp_df.dropna(
        subset=["Date"]
    )

    result = (
        temp_df
        .set_index("Date")
        .resample("ME")[value_column]
        .sum()
        .reset_index()
    )

    result["Month"] = (
        result["Date"]
        .dt.strftime("%Y-%m")
    )

    result = result[
        [
            "Month",
            value_column,
        ]
    ]

    return result


# ============================================================
# VALUE COUNTS
# ============================================================

def value_counts(
    df,
    column
):
    """
    Count the frequency of each value
    in a column.
    """

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
        "Count",
    ]

    return result


# ============================================================
# GROUP + COUNT
# ============================================================

def group_and_count(
    df,
    group_column
):
    """
    Count records for each group.
    """

    validate_column(
        df,
        group_column
    )

    result = (
        df[group_column]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    result.columns = [
        group_column,
        "Count",
    ]

    return result
# ============================================================
# FILTER DATA
# ============================================================

def filter_dataframe(
    df,
    filter_column,
    filter_value
):
    """
    Filter dataframe using an exact value match.
    """

    validate_column(
        df,
        filter_column
    )

    # Convert both sides to strings for a flexible
    # comparison across CSV/Excel datasets.
    mask = (
        df[filter_column]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        str(filter_value)
        .strip()
        .lower()
    )

    filtered_df = df.loc[
        mask
    ].copy()

    if filtered_df.empty:

        raise ValueError(
            f"No rows found where "
            f"{filter_column} = {filter_value}"
        )

    return filtered_df


# ============================================================
# FILTERED SUM
# ============================================================

def filtered_sum(
    df,
    filter_column,
    filter_value,
    value_column
):
    """
    Filter the dataset and calculate a sum.
    """

    filtered_df = filter_dataframe(
        df,
        filter_column,
        filter_value
    )

    return calculate_sum(
        filtered_df,
        value_column
    )


# ============================================================
# FILTERED AVERAGE
# ============================================================

def filtered_average(
    df,
    filter_column,
    filter_value,
    value_column
):
    """
    Filter the dataset and calculate an average.
    """

    filtered_df = filter_dataframe(
        df,
        filter_column,
        filter_value
    )

    return calculate_average(
        filtered_df,
        value_column
    )


# ============================================================
# FILTERED COUNT
# ============================================================

def filtered_count(
    df,
    filter_column,
    filter_value,
    count_column=None
):
    """
    Filter the dataset and count rows/non-null values.
    """

    filtered_df = filter_dataframe(
        df,
        filter_column,
        filter_value
    )

    return calculate_count(
        filtered_df,
        count_column
    )


# ============================================================
# FILTERED UNIQUE COUNT
# ============================================================

def filtered_unique_count(
    df,
    filter_column,
    filter_value,
    value_column
):
    """
    Filter the dataset and calculate the number
    of unique values.
    """

    filtered_df = filter_dataframe(
        df,
        filter_column,
        filter_value
    )

    return calculate_unique_count(
        filtered_df,
        value_column
    )