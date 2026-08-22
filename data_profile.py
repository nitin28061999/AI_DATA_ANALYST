import contextlib
import pandas as pd
import pandas as pd


# ============================================================
# BASIC COLUMN CLASSIFICATION
# ============================================================

def _is_id_like(column_name, series):
    """
    Detect columns that are likely identifiers.
    """

    name = str(column_name).lower()

    id_keywords = [
        "id",
        "code",
        "number",
        "no",
        "number",
        "invoice",
        "employee",
        "customer",
        "student",
        "order",
        "transaction",
        "account",
    ]

    if any(keyword in name for keyword in id_keywords):
        return True

    # Very high cardinality object columns are often IDs.
    if series.dtype == "object":
        unique_ratio = (
            series.nunique(dropna=True) / max(len(series), 1)
        )

        if unique_ratio > 0.95:
            return True

    return False


def _classify_column(column_name, series):
    """
    Classify a dataframe column.

    Possible kinds:

    numeric
    datetime
    categorical
    boolean
    text
    id
    """

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):

        return "id" if _is_id_like(column_name, series) else "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if _is_id_like(column_name, series):
        return "id"

    # Try detecting dates stored as strings.
    if series.dtype == "object":

        sample = series.dropna().head(100)

        if len(sample) > 0:

            parsed = pd.to_datetime(
                sample,
                errors="coerce"
            )

            valid_ratio = parsed.notna().mean()

            if valid_ratio >= 0.8:
                return "datetime"

    # Low-cardinality text is usually categorical.
    if series.dtype == "object":

        unique_count = series.nunique(
            dropna=True
        )

        return "categorical" if unique_count <= 100 else "text"
    return "text"


# ============================================================
# SAMPLE VALUES
# ============================================================
def _get_sample_values(series, limit=8):
    """
    Return representative non-null sample values.
    """

    values = (
        series
        .dropna()
        .drop_duplicates()
        .head(limit)
        .tolist()
    )

    cleaned = []

    for value in values:

        if isinstance(
            value,
            (pd.Timestamp,)
        ):
            value = value.isoformat()

        elif hasattr(
            value,
            "item"
        ):

            with contextlib.suppress(Exception):
                value = value.item()
        cleaned.append(
            value
        )

    return cleaned
# ============================================================
# NUMERIC SUMMARY
# ============================================================

def _numeric_summary(series):
    """
    Create a small numeric summary.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if values.empty:
        return {}

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


# ============================================================
# COLUMN PROFILE
# ============================================================

def profile_column(
    column_name,
    series
):
    """
    Create detailed metadata for one column.
    """

    kind = _classify_column(
        column_name,
        series
    )

    profile = {
        "name": str(column_name),
        "dtype": str(series.dtype),
        "kind": kind,
        "row_count": len(series),
        "missing_count": int(series.isna().sum()),
        "missing_percentage": round(float(series.isna().mean() * 100), 2),
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": _get_sample_values(series),
    }

    if kind == "numeric":

        profile[
            "numeric_summary"
        ] = _numeric_summary(
            series
        )

    return profile


# ============================================================
# DATASET PROFILE
# ============================================================

def create_profile(df):
    """
    Create a comprehensive profile of the uploaded dataset.

    This function is intentionally dataset-agnostic.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    columns = [profile_column(column, df[column]) for column in df.columns]
    numeric_columns = [
        item["name"]
        for item in columns
        if item["kind"] == "numeric"
    ]

    categorical_columns = [
        item["name"]
        for item in columns
        if item["kind"] == "categorical"
    ]

    datetime_columns = [
        item["name"]
        for item in columns
        if item["kind"] == "datetime"
    ]

    id_columns = [
        item["name"]
        for item in columns
        if item["kind"] == "id"
    ]

    text_columns = [
        item["name"]
        for item in columns
        if item["kind"] == "text"
    ]

    boolean_columns = [
        item["name"]
        for item in columns
        if item["kind"] == "boolean"
    ]

    total_missing = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "column_names": [str(column) for column in df.columns],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "id_columns": id_columns,
        "text_columns": text_columns,
        "boolean_columns": boolean_columns,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_profile(df):
    """
    Backward-compatible alias.
    """

    return create_profile(df)