import contextlib
import re

import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

_ID_NAME_KEYWORDS = (
    "id",
    "code",
    "number",
    "no",
    "invoice",
    "employee",
    "customer",
    "student",
    "order",
    "transaction",
    "account",
    "reference",
)

_DATE_NAME_KEYWORDS = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "created",
    "updated",
    "modified",
)

_ID_PATTERN = re.compile(
    r"^[A-Za-z]+\d+$|^\d+[A-Za-z]+$|^[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*$"
)


# ============================================================
# BASIC COLUMN CLASSIFICATION
# ============================================================


def _is_id_like(column_name, series):
    """
    Detect columns that are likely identifiers.

    Detection uses:
    - Explicit identifier-related column names.
    - Highly unique alphanumeric values.
    - Numeric columns with identifier-like names.
    """

    name = str(column_name).strip().lower()

    # Strong name-based detection.
    if any(keyword in name for keyword in _ID_NAME_KEYWORDS):
        return True

    non_null = series.dropna()

    if non_null.empty:
        return False

    unique_ratio = (
        non_null.nunique(dropna=True)
        / max(len(non_null), 1)
    )

    # High-cardinality text identifiers such as:
    # A001, A002, A003, ...
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        if unique_ratio >= 0.95 and len(non_null) >= 5:
            values = non_null.astype(str).str.strip()

            if values.map(
                lambda value: bool(_ID_PATTERN.match(value))
            ).mean() >= 0.8:
                return True

    return False


def _looks_like_datetime(series):
    """
    Detect datetime values stored as strings.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    sample = non_null.head(100)

    parsed = pd.to_datetime(
        sample,
        errors="coerce",
        format="mixed",
    )

    valid_ratio = float(parsed.notna().mean())

    return valid_ratio >= 0.8


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

    name = str(column_name).strip().lower()

    # Boolean must be checked before numeric.
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # Native datetime columns.
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Date-like column names get an early datetime check.
    if (
        any(keyword in name for keyword in _DATE_NAME_KEYWORDS)
        and (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )
    ):
        if _looks_like_datetime(series):
            return "datetime"

    # Numeric columns.
    if pd.api.types.is_numeric_dtype(series):
        return "id" if _is_id_like(column_name, series) else "numeric"
    # Detect string dates even when the column name is generic.
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        if _looks_like_datetime(series):
            return "datetime"

    # Identifier-like columns.
    if _is_id_like(column_name, series):
        return "id"

    # Text / categorical classification.
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        unique_count = int(
            series.nunique(dropna=True)
        )

        # A low-cardinality string column is categorical.
        #
        # This intentionally does not depend on dataset size.
        # For example:
        # ["North", "South", "North", "East"]
        # is categorical even though the dataset has only 4 rows.
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

        # Convert pandas timestamps to strings.
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()

        # Convert NumPy scalar values to native Python values.
        elif hasattr(value, "item"):
            with contextlib.suppress(Exception):
                value = value.item()

        cleaned.append(value)

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
        errors="coerce",
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


def profile_column(column_name, series):
    """
    Create detailed metadata for one column.
    """

    kind = _classify_column(
        column_name,
        series,
    )

    profile = {
        "name": str(column_name),
        "dtype": str(series.dtype),
        "kind": kind,
        "row_count": len(series),
        "missing_count": int(
            series.isna().sum()
        ),
        "missing_percentage": round(
            float(series.isna().mean() * 100),
            2,
        ),
        "unique_count": int(
            series.nunique(dropna=True)
        ),
        "sample_values": _get_sample_values(
            series
        ),
    }

    if kind == "numeric":
        profile["numeric_summary"] = _numeric_summary(
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

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    columns = [
        profile_column(
            column,
            df[column],
        )
        for column in df.columns
    ]

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
        "column_names": [
            str(column)
            for column in df.columns
        ],
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
    Backward-compatible alias for create_profile().
    """

    return create_profile(df)
