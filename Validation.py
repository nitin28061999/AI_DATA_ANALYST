import pandas as pd


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_column(df, column):
    """
    Validate that a column exists.
    """

    if column not in df.columns:

        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    return True


# ============================================================
# NUMERIC COLUMN VALIDATION
# ============================================================

def validate_numeric_column(df, column):
    """
    Validate that a column exists and contains
    usable numeric data.
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
            f"Column '{column}' does not contain "
            "usable numeric values."
        )

    return True


# ============================================================
# FILTER VALIDATION
# ============================================================

def validate_filters(df, filters):
    """
    Validate multiple filter conditions.
    """

    if not isinstance(filters, list):

        raise ValueError(
            "'filters' must be a list."
        )

    for filter_item in filters:

        if not isinstance(
            filter_item,
            dict
        ):

            raise ValueError(
                "Each filter must be an object."
            )

        column = filter_item.get(
            "column"
        )

        if not column:

            raise ValueError(
                "Filter is missing 'column'."
            )

        validate_column(
            df,
            column
        )

        if "value" not in filter_item:

            raise ValueError(
                f"Filter for '{column}' "
                "is missing 'value'."
            )

    return True