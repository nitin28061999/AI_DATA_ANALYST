# analyst.py

from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis_tools import (
    calculate_sum,
    calculate_average,
    calculate_count,
    calculate_unique_count,
    calculate_min,
    calculate_max,
    group_and_sum,
    group_and_average,
    group_and_count,
    top_n,
    percentage_of_total,
    monthly_sum,
    value_counts,
    filtered_sum,
    filtered_average,
    filtered_count,
    filtered_unique_count,
    filtered_min,
    filtered_max,
    filtered_group_and_sum,
    filtered_group_and_average,
    filtered_value_counts,
    filtered_top_n,
)


# ============================================================
# OPTIONAL GEMINI
# ============================================================
#
# Gemini is OFF by default.
#
# Normal supported questions make ZERO API calls.
#
# To enable Gemini fallback:
#
# ANALYST_USE_GEMINI=1
# GEMINI_API_KEY=your_key
#
# Optional:
#
# GEMINI_MODEL=gemini-3.6-flash
#
# ============================================================

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

USE_GEMINI = (
    os.getenv(
        "ANALYST_USE_GEMINI",
        "0",
    )
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

client = None

if USE_GEMINI:
    try:
        from google import genai

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if api_key:
            client = genai.Client(
                api_key=api_key
            )
        else:
            USE_GEMINI = False

    except Exception:
        USE_GEMINI = False
        client = None


# ============================================================
# SUPPORTED OPERATIONS
# ============================================================

SUPPORTED_OPERATIONS = {
    "calculate_sum",
    "calculate_average",
    "calculate_count",
    "calculate_unique_count",
    "calculate_min",
    "calculate_max",
    "group_and_sum",
    "group_and_average",
    "group_and_count",
    "top_n",
    "percentage_of_total",
    "monthly_sum",
    "value_counts",
    "filtered_sum",
    "filtered_average",
    "filtered_count",
    "filtered_unique_count",
    "filtered_min",
    "filtered_max",
    "filtered_group_and_sum",
    "filtered_group_and_average",
    "filtered_value_counts",
    "filtered_top_n",
}


COLUMN_OPS = {
    "calculate_sum",
    "calculate_average",
    "calculate_count",
    "calculate_unique_count",
    "calculate_min",
    "calculate_max",
    "value_counts",
}


GROUP_VALUE_OPS = {
    "group_and_sum",
    "group_and_average",
    "top_n",
    "percentage_of_total",
    "filtered_group_and_sum",
    "filtered_group_and_average",
    "filtered_top_n",
}


FILTERED_OPS = {
    "filtered_sum",
    "filtered_average",
    "filtered_count",
    "filtered_unique_count",
    "filtered_min",
    "filtered_max",
    "filtered_group_and_sum",
    "filtered_group_and_average",
    "filtered_value_counts",
    "filtered_top_n",
}


FILTER_OPERATORS = {
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


# ============================================================
# GENERAL HELPERS
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
) -> None:
    """Validate that df is a usable pandas DataFrame."""

    if df is None:
        raise ValueError(
            "DataFrame cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "The dataset has no columns."
        )


def get_dataset_columns(
    df: pd.DataFrame,
) -> List[str]:
    """Return dataset column names as strings."""

    validate_dataframe(df)

    return [
        str(column)
        for column in df.columns
    ]


def _norm(
    value: Any,
) -> str:
    """Normalize text for matching."""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .strip()
        .lower(),
    ).strip()


def _safe(
    value: Any,
) -> Any:
    """
    Convert pandas/numpy values into
    JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _safe(item)
            for item in value
        ]

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return [
            _safe(record)
            for record in value.to_dict(
                orient="records"
            )
        ]

    if isinstance(
        value,
        pd.Series,
    ):
        return {
            str(key): _safe(item)
            for key, item in value.to_dict().items()
        }

    if hasattr(
        value,
        "item",
    ):
        with contextlib.suppress(
            Exception
        ):
            return _safe(
                value.item()
            )

    if hasattr(
        value,
        "isoformat",
    ):
        with contextlib.suppress(
            Exception
        ):
            return value.isoformat()

    try:
        json.dumps(value)
        return value

    except Exception:
        return str(value)


def serialize_result(
    result: Any,
) -> Any:
    """Public result serialization helper."""

    return _safe(result)


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(
    text: str,
) -> Dict[str, Any]:
    """Extract a JSON object from model output."""

    if not text:
        raise ValueError(
            "Empty JSON response."
        )

    text = text.strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```",
        "",
        text,
    )

    text = text.strip()

    with contextlib.suppress(
        json.JSONDecodeError
    ):
        value = json.loads(text)

        if isinstance(
            value,
            dict,
        ):
            return value

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError(
            "No JSON object found in model response."
        )

    json_text = text[
        start : end + 1
    ]

    try:
        value = json.loads(
            json_text
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not parse model JSON: "
            f"{exc}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Analysis plan must be a JSON object."
        )

    return value


# ============================================================
# PROFILE
# ============================================================

def build_profile(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build a profile from the real DataFrame."""

    validate_dataframe(df)

    columns = get_dataset_columns(df)

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    datetime_columns = [
        column
        for column in columns
        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        )
    ]

    text_columns = [
        column
        for column in columns
        if column not in numeric_columns
        and column not in datetime_columns
    ]

    return {
        "columns": columns,
        "row_count": len(df),
        "numeric_columns": numeric_columns,
        "datetime_columns": datetime_columns,
        "text_columns": text_columns,
    }


def normalize_profile(
    df: pd.DataFrame,
    profile: Optional[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Normalize a supplied profile.

    The real DataFrame is authoritative.
    """

    validate_dataframe(df)

    generated = build_profile(df)

    if profile is None:
        return generated

    if not isinstance(
        profile,
        dict,
    ):
        raise ValueError(
            "Dataset profile must be a dictionary."
        )

    result = dict(profile)

    # Actual DataFrame columns always win.
    result["columns"] = generated[
        "columns"
    ]

    for key, value in generated.items():
        result.setdefault(
            key,
            value,
        )

    return result


def _profile_columns(
    profile: Optional[
        Dict[str, Any]
    ],
) -> List[str]:
    """Return normalized profile columns."""

    if profile is None:
        return []

    if not isinstance(
        profile,
        dict,
    ):
        raise ValueError(
            "Dataset profile must be a dictionary."
        )

    columns = profile.get(
        "columns",
        [],
    )

    if isinstance(
        columns,
        dict,
    ):
        columns = list(
            columns.keys()
        )

    if isinstance(
        columns,
        (list, tuple),
    ):
        return [
            str(column)
            for column in columns
        ]

    return []


# ============================================================
# COLUMN SELECTION
# ============================================================

def _find_column(
    question: str,
    columns: List[str],
) -> Optional[str]:
    """
    Find an explicitly mentioned column.

    Longest names are checked first.
    """

    q = _norm(question)

    return next(
        (
            column
            for column in sorted(
                columns,
                key=len,
                reverse=True,
            )
            if _norm(column) in q
        ),
        None,
    )


def _semantic_column(
    columns: List[str],
    keywords: List[str],
) -> Optional[str]:
    """Find the best semantic column match."""

    scored = []

    normalized_keywords = [
        _norm(keyword)
        for keyword in keywords
    ]

    for column in columns:
        name = _norm(column)
        score = 0

        for keyword in normalized_keywords:

            if name == keyword:
                score += 100

            elif keyword in name:
                score += 20

        if score:
            scored.append(
                (
                    score,
                    -len(column),
                    column,
                )
            )

    return None if not scored else max(scored)[2]


def _numeric_columns(
    df: pd.DataFrame,
) -> List[str]:
    """Return numeric columns."""

    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


def _text_columns(
    df: pd.DataFrame,
) -> List[str]:
    """Return non-numeric columns."""

    numeric = set(
        _numeric_columns(df)
    )

    return [
        str(column)
        for column in df.columns
        if str(column) not in numeric
    ]


def _value_column(
    question: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """Choose the numeric value column."""

    columns = get_dataset_columns(df)
    numeric = _numeric_columns(df)

    explicit = _find_column(
        question,
        columns,
    )

    if explicit in numeric:
        return explicit

    found = _semantic_column(
        numeric,
        [
            "revenue",
            "sales",
            "sale",
            "amount",
            "price",
            "profit",
            "income",
            "expense",
            "expenses",
            "salary",
            "score",
            "cost",
            "quantity",
            "units",
            "value",
            "total",
        ],
    )

    if found:
        return found

    return numeric[0] if numeric else None


def _group_column(
    question: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """Choose the grouping column."""

    columns = get_dataset_columns(df)

    explicit = _find_column(
        question,
        columns,
    )

    if explicit:
        return explicit

    text = _text_columns(df)

    found = _semantic_column(
        text,
        [
            "city",
            "state",
            "country",
            "region",
            "product",
            "category",
            "department",
            "employee",
            "customer",
            "status",
            "brand",
            "segment",
            "type",
            "location",
        ],
    )

    if found:
        return found

    return text[0] if text else None


def _count_column(
    question: str,
    df: pd.DataFrame,
) -> str:
    """Choose the column whose rows should be counted."""

    columns = get_dataset_columns(df)
    q = _norm(question)

    candidates = [
        (
            "invoice",
            [
                "invoice",
                "invoice_id",
                "invoice id",
            ],
        ),
        (
            "order",
            [
                "order",
                "order_id",
                "order id",
            ],
        ),
        (
            "transaction",
            [
                "transaction",
                "transaction_id",
                "transaction id",
            ],
        ),
        (
            "customer",
            [
                "customer",
                "customer_id",
                "customer id",
            ],
        ),
        (
            "employee",
            [
                "employee",
                "employee_id",
                "employee id",
            ],
        ),
    ]

    for trigger, keys in candidates:

        if trigger not in q:
            continue

        found = _semantic_column(
            columns,
            keys,
        )

        if found:
            return found

    explicit = _find_column(
        question,
        columns,
    )

    return explicit or columns[0]


def _unique_column(
    question: str,
    df: pd.DataFrame,
) -> str:
    """Choose the column whose distinct values are counted."""

    columns = get_dataset_columns(df)
    q = _norm(question)

    candidates = [
        (
            "customer",
            [
                "customer",
                "customer_id",
                "customer id",
            ],
        ),
        (
            "product",
            [
                "product",
                "product_id",
                "product id",
            ],
        ),
        (
            "employee",
            [
                "employee",
                "employee_id",
                "employee id",
            ],
        ),
        (
            "invoice",
            [
                "invoice",
                "invoice_id",
                "invoice id",
            ],
        ),
        (
            "order",
            [
                "order",
                "order_id",
                "order id",
            ],
        ),
        (
            "transaction",
            [
                "transaction",
                "transaction_id",
                "transaction id",
            ],
        ),
        (
            "city",
            [
                "city",
            ],
        ),
    ]

    for trigger, keys in candidates:

        if trigger not in q:
            continue

        found = _semantic_column(
            columns,
            keys,
        )

        if found:
            return found

    explicit = _find_column(
        question,
        columns,
    )

    return explicit or columns[0]


# ============================================================
# FILTER HELPERS
# ============================================================

def _coerce(
    value: Any,
    series: pd.Series,
) -> Any:
    """Coerce text into the column's data type."""

    if not isinstance(
        value,
        str,
    ):
        return value

    value = (
        value
        .strip()
        .strip("\"'")
        .rstrip(".,;!?")
        .strip()
    )

    if pd.api.types.is_numeric_dtype(
        series
    ):
        with contextlib.suppress(
            ValueError
        ):
            return float(value)

        with contextlib.suppress(
            ValueError
        ):
            return int(value)

    if pd.api.types.is_bool_dtype(
        series
    ):
        lowered = value.lower()

        if lowered in {
            "true",
            "yes",
        }:
            return True

        if lowered in {
            "false",
            "no",
        }:
            return False

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        with contextlib.suppress(
            Exception
        ):
            return pd.to_datetime(
                value
            )

    return value


def _dataset_value(
    column: str,
    value: Any,
    df: pd.DataFrame,
) -> Any:
    """
    Resolve a textual value against an
    actual dataset value.

    Example:
        delhi -> Delhi
        laptop -> Laptop
    """

    if not isinstance(
        value,
        str,
    ):
        return value

    value = (
        value
        .strip()
        .strip("\"'")
        .rstrip(".,;!?")
        .strip()
    )

    series = df[column]

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mask = (
        normalized
        == value.lower()
    )

    return (
        series[mask].iloc[0]
        if mask.any()
        else _coerce(
            value,
            series,
        )
    )

def _find_column_for_value(
    value: str,
    df: pd.DataFrame,
    preferred_keywords: Optional[
        List[str]
    ] = None,
) -> Optional[str]:
    """
    Find a dataset column containing
    an exact textual value.
    """

    value_norm = _norm(value)

    columns = get_dataset_columns(df)

    preferred = (
        preferred_keywords or []
    )

    ordered_columns = sorted(
        columns,
        key=lambda column: (
            0
            if any(
                _norm(keyword)
                in _norm(column)
                for keyword in preferred
            )
            else 1,
            len(column),
        ),
    )

    for column in ordered_columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if value_norm in set(values):
            return column

    return None


# ============================================================
# FILTER EXTRACTION
# ============================================================

def _extract_filters(
    question: str,
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Extract deterministic filters.

    Supported examples:

        City = Delhi
        City is Delhi
        City == Delhi
        City != Delhi
        Price > 500
        Price >= 500
        Price < 500
        Price <= 500
        Product contains Laptop
        Salary between 50000 and 70000
        in Delhi
        for Laptop
        from Delhi
        City Delhi
        Product Laptop
    """

    columns = get_dataset_columns(df)
    q = question.strip()

    filters: List[
        Dict[str, Any]
    ] = []

    used_columns: set[str] = set()

    def add(
        column: str,
        operator: str,
        value: Any,
    ) -> None:

        if column in used_columns:
            return

        if isinstance(
            value,
            str,
        ):
            value = _dataset_value(
                column,
                value,
                df,
            )

        if operator == "==":
            operator = "="

        filters.append(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

        used_columns.add(column)

    # --------------------------------------------------------
    # Explicit column + BETWEEN
    # --------------------------------------------------------

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s+between\s+"
            rf"(.+?)"
            rf"\s+and\s+"
            rf"(.+?)"
            rf"(?=\s+(?:and|for|in|from|where)\b|[,;?]|$)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        first_value = match[1].strip()

        second_value = match[2].strip()

        add(
            column,
            "between",
            [
                _dataset_value(
                    column,
                    first_value,
                    df,
                ),
                _dataset_value(
                    column,
                    second_value,
                    df,
                ),
            ],
        )

    # --------------------------------------------------------
    # Explicit column + contains/is/operators
    # --------------------------------------------------------

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s*"
            rf"(>=|<=|!=|==|=|>|<|contains|is)"
            rf"\s*"
            rf"(.+?)"
            rf"(?=\s+(?:and|for|in|from|where)\b|[,;?]|$)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        operator = match[1].lower().strip()

        raw_value = match[2].strip()

        if operator == "is":
            operator = "="

        if operator == "contains":
            raw_value = (
                raw_value
                .strip("\"'")
                .strip()
            )

            add(
                column,
                "contains",
                raw_value,
            )

        else:
            add(
                column,
                operator,
                raw_value,
            )

    # --------------------------------------------------------
    # Explicit numeric comparisons
    #
    # Handles:
    # Price > 500
    # Salary >= 50000
    # Revenue < 1000
    # --------------------------------------------------------

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s*"
            rf"(>=|<=|!=|>|<)"
            rf"\s*"
            rf"(-?\d+(?:\.\d+)?)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        add(column, match[1], _coerce(match[2], df[column]))

    # --------------------------------------------------------
    # Generic numeric BETWEEN
    #
    # Example:
    # revenue between 100 and 500
    # --------------------------------------------------------

    generic_between = re.search(
        r"\bbetween\s+"
        r"(-?\d+(?:\.\d+)?)"
        r"\s+and\s+"
        r"(-?\d+(?:\.\d+)?)",
        q,
        re.IGNORECASE,
    )

    if generic_between:

        value_column = _value_column(
            q,
            df,
        )

        if (
            value_column
            and value_column
            not in used_columns
        ):
            add(
                value_column,
                "between",
                [
                    _coerce(generic_between[1], df[value_column]),
                    _coerce(generic_between[2], df[value_column]),
                ],
            )

    # --------------------------------------------------------
    # Natural equality:
    #
    # in Delhi
    # for Laptop
    # from Delhi
    # --------------------------------------------------------

    natural_pattern = re.compile(
        r"\b(in|for|from)\s+"
        r"['\"]?"
        r"([^,;?.]+?)"
        r"['\"]?"
        r"(?=\s+(?:for|in|from|and|where)\b|[,;?.]|$)",
        re.IGNORECASE,
    )

    natural_matches = (
        natural_pattern.findall(q)
    )

    for keyword, raw_value in natural_matches:

        value = (
            raw_value
            .strip()
            .strip("\"'")
            .strip()
        )

        if not value:
            continue

        if keyword.lower() in {
            "in",
            "from",
        }:
            preferred_keywords = [
                "city",
                "state",
                "country",
                "region",
                "location",
            ]

        else:
            preferred_keywords = [
                "product",
                "category",
                "brand",
                "type",
                "segment",
                "department",
            ]

        column = _find_column_for_value(
            value,
            df,
            preferred_keywords,
        )

        if column:
            add(
                column,
                "=",
                value,
            )

    # --------------------------------------------------------
    # Natural equality:
    #
    # City Delhi
    # Product Laptop
    # Category Electronics
    # --------------------------------------------------------

    operation_words = {
        "revenue",
        "sales",
        "amount",
        "profit",
        "average",
        "avg",
        "mean",
        "total",
        "sum",
        "count",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "largest",
        "smallest",
    }

    for column in columns:

        if column in used_columns:
            continue

        escaped = re.escape(
            column
        )

        pattern = (
            rf"(?<!\w){escaped}(?!\w)"
            rf"\s+"
            rf"['\"]?"
            rf"([^,;?.]+?)"
            rf"['\"]?"
            rf"(?=\s+(?:and|for|in|from|where)\b|[,;?.]|$)"
        )

        match = re.search(
            pattern,
            q,
            re.IGNORECASE,
        )

        if not match:
            continue

        candidate = match[1].strip().strip("\"'").strip()

        if not candidate:
            continue

        if _norm(candidate) in operation_words:
            continue

        dataset_values = set(
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if _norm(candidate) in dataset_values:
            add(
                column,
                "=",
                candidate,
            )

    return filters


# ============================================================
# FILTER NORMALIZATION
# ============================================================

def normalize_filters(
    filters: Any,
    df: Optional[
        pd.DataFrame
    ] = None,
) -> List[Dict[str, Any]]:
    """
    Validate and normalize filters.
    """

    if not isinstance(
        filters,
        list,
    ):
        raise ValueError(
            "Filters must be a list."
        )

    if not filters:
        raise ValueError(
            "At least one filter is required."
        )

    result: List[
        Dict[str, Any]
    ] = []

    for index, item in enumerate(
        filters
    ):

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Filter #{index + 1} "
                "must be a dictionary."
            )

        if "column" not in item:
            raise ValueError(
                f"Filter #{index + 1} "
                "is missing 'column'."
            )

        if "value" not in item:
            raise ValueError(
                f"Filter #{index + 1} "
                "is missing 'value'."
            )

        column = item[
            "column"
        ]

        if not column:
            raise ValueError(
                f"Filter #{index + 1} "
                "has an empty column."
            )

        operator = str(
            item.get(
                "operator",
                "=",
            )
            or "="
        ).strip().lower()

        if operator == "==":
            operator = "="

        if operator not in FILTER_OPERATORS:
            raise ValueError(
                f"Unsupported filter "
                f"operator '{operator}'."
            )

        value = item[
            "value"
        ]

        if operator == "between":

            if not isinstance(
                value,
                (list, tuple),
            ):
                raise ValueError(
                    "The 'between' filter "
                    "requires exactly two values."
                )

            if len(value) != 2:
                raise ValueError(
                    "The 'between' filter "
                    "requires exactly two values."
                )

            value = [
                value[0],
                value[1],
            ]

        if (
            df is not None
            and column in df.columns
        ):

            if operator == "between":

                value = [
                    _coerce(
                        value[0],
                        df[column],
                    ),
                    _coerce(
                        value[1],
                        df[column],
                    ),
                ]

            elif operator == "contains":

                value = str(
                    value
                ).strip()

            else:

                value = _coerce(
                    value,
                    df[column],
                )

        result.append(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

    return result


# ============================================================
# PLAN VALIDATION
# ============================================================

def validate_plan_columns(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> None:
    """
    Validate every column referenced by
    the analysis plan.
    """

    validate_dataframe(df)

    columns = set(
        get_dataset_columns(df)
    )

    operation = plan.get(
        "operation"
    )

    def require(
        key: str,
    ) -> None:

        value = plan.get(key)

        if not value:
            raise ValueError(
                f"{operation} requires "
                f"'{key}'."
            )

        if value not in columns:
            raise ValueError(
                f"Column '{value}' does not exist. "
                f"Available columns: "
                f"{sorted(columns)}"
            )

    if operation in COLUMN_OPS:
        require("column")

    if operation in GROUP_VALUE_OPS:
        require("group_column")
        require("value_column")

    if operation == "group_and_count":
        require("group_column")

    if operation == "monthly_sum":
        require("date_column")
        require("value_column")

    if operation in FILTERED_OPS:

        filters = plan.get(
            "filters"
        )

        if (
            not isinstance(
                filters,
                list,
            )
            or not filters
        ):
            raise ValueError(
                f"{operation} requires "
                "'filters'."
            )

        for index, filter_item in enumerate(
            filters
        ):

            if not isinstance(
                filter_item,
                dict,
            ):
                raise ValueError(
                    f"Filter #{index + 1} "
                    "must be a dictionary."
                )

            filter_column = (
                filter_item.get(
                    "column"
                )
            )

            if not filter_column:
                raise ValueError(
                    f"Filter #{index + 1} "
                    "is missing 'column'."
                )

            if (
                filter_column
                not in columns
            ):
                raise ValueError(
                    f"Filter column "
                    f"'{filter_column}' "
                    "does not exist."
                )

            if "value" not in filter_item:
                raise ValueError(
                    f"Filter #{index + 1} "
                    "is missing 'value'."
                )

        if operation in {
            "filtered_sum",
            "filtered_average",
            "filtered_unique_count",
            "filtered_min",
            "filtered_max",
            "filtered_group_and_sum",
            "filtered_group_and_average",
            "filtered_top_n",
        }:
            require("value_column")

        if operation == "filtered_count":
            require("count_column")

        if operation in {
            "filtered_group_and_sum",
            "filtered_group_and_average",
            "filtered_top_n",
        }:
            require("group_column")

        if operation == "filtered_value_counts":
            require("column")


def validate_plan(
    plan: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate an analysis plan against a profile."""

    if not isinstance(
        plan,
        dict,
    ):
        raise ValueError(
            "Analysis plan must be a dictionary."
        )

    operation = str(
        plan.get(
            "operation",
            "",
        )
    ).strip()

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation "
            f"'{operation}'."
        )

    result = dict(plan)

    result[
        "operation"
    ] = operation

    if operation.startswith(
        "filtered_"
    ):
        result[
            "filters"
        ] = normalize_filters(
            result.get(
                "filters"
            )
        )

    if operation in {
        "top_n",
        "filtered_top_n",
    }:

        try:
            n = int(
                result.get(
                    "n",
                    5,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "n must be an integer."
            ) from exc

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        result["n"] = n

    columns = _profile_columns(
        profile
    )

    if columns:

        for key in (
            "column",
            "group_column",
            "value_column",
            "count_column",
            "date_column",
        ):

            value = result.get(
                key
            )

            if (
                value
                and value not in columns
            ):
                raise ValueError(
                    f"Column '{value}' "
                    "does not exist in the dataset."
                )

        for filter_item in result.get(
            "filters",
            [],
        ):

            if (
                filter_item["column"]
                not in columns
            ):
                raise ValueError(
                    f"Filter column "
                    f"'{filter_item['column']}' "
                    "does not exist."
                )

    return result


def normalize_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and normalize a plan
    against the real DataFrame.
    """

    validate_dataframe(df)

    if not isinstance(
        plan,
        dict,
    ):
        raise ValueError(
            "Analysis plan must be a dictionary."
        )

    operation = str(
        plan.get(
            "operation",
            "",
        )
    ).strip()

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation "
            f"'{operation}'."
        )

    result = dict(plan)

    result[
        "operation"
    ] = operation

    if operation.startswith(
        "filtered_"
    ):
        result[
            "filters"
        ] = normalize_filters(
            result.get(
                "filters"
            ),
            df=df,
        )

    if operation in {
        "top_n",
        "filtered_top_n",
    }:

        try:
            n = int(
                result.get(
                    "n",
                    5,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "n must be an integer."
            ) from exc

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        result["n"] = n

    validate_plan_columns(
        df,
        result,
    )

    return result


# ============================================================
# TOP N EXTRACTION
# ============================================================

def _extract_n(
    question: str,
    default: int = 5,
) -> int:
    """Extract N from top/first/best/highest/bottom N."""

    match = re.search(
        r"\b(?:top|first|best|highest|bottom)\s+(\d+)\b",
        question,
        re.IGNORECASE,
    )

    return default if not match else max(1, int(match[1]))


def _has_grouping_intent(
    question: str,
) -> bool:
    """Determine whether the question requests grouped output."""

    q = _norm(question)

    return any(
        phrase in q
        for phrase in (
            " by ",
            "group by",
            "grouped by",
            "per city",
            "per product",
            "per category",
            "per department",
            "per region",
            "per state",
            "per country",
        )
    )


# ============================================================
# DETERMINISTIC PLANNER
# ============================================================

def deterministic_plan(
    question: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:  # sourcery skip: low-code-quality
    """
    Create an analysis plan without
    using an external API.
    """

    validate_dataframe(df)

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    q = _norm(question)

    filters = _extract_filters(
        question,
        df,
    )

    filtered = bool(filters)

    grouping_intent = (
        _has_grouping_intent(
            question
        )
    )

    # --------------------------------------------------------
    # TOP N / HIGHEST / BEST
    # --------------------------------------------------------

    explicit_top = (
        re.search(
            r"\btop\s+\d+\b",
            q,
        )
        is not None
    )

    top_words = (
        "highest",
        "largest",
        "greatest",
        "best performing",
        "top ",
        "maximum by",
    )

    top_intent = (
        explicit_top
        or any(
            phrase in q
            for phrase in top_words
        )
    )

    # "highest revenue" by itself is a maximum.
    # "highest revenue by city" is top-N grouped.
    if top_intent and (
        grouping_intent
        or explicit_top
        or " by " in q
    ):

        group = _group_column(
            question,
            df,
        )

        value = _value_column(
            question,
            df,
        )

        if group and value:

            return {
                "operation": (
                    "filtered_top_n"
                    if filtered
                    else "top_n"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                "group_column": group,
                "value_column": value,
                "n": _extract_n(
                    question,
                    1,
                ),
            }

    # --------------------------------------------------------
    # PERCENTAGE / SHARE
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "percentage of total",
            "percent of total",
            "contribution",
            "share of",
            "share ",
        )
    ):

        group = _group_column(
            question,
            df,
        )

        value = _value_column(
            question,
            df,
        )

        if (
            group
            and value
            and not filtered
        ):
            return {
                "operation":
                    "percentage_of_total",
                "group_column": group,
                "value_column": value,
            }

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "monthly",
            "by month",
            "per month",
            "month wise",
            "month-wise",
            "monthly trend",
            "monthly revenue",
            "monthly sales",
            "sales by month",
            "revenue by month",
        )
    ):

        date = _semantic_column(
            get_dataset_columns(df),
            [
                "date",
                "datetime",
                "timestamp",
            ],
        )

        value = _value_column(
            question,
            df,
        )

        if date and value:

            return {
                "operation": "monthly_sum",
                "date_column": date,
                "value_column": value,
            }

    # --------------------------------------------------------
    # UNIQUE / DISTINCT
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "unique",
            "distinct",
            "different ",
        )
    ):

        column = _unique_column(
            question,
            df,
        )

        if filtered:

            return {
                "operation":
                    "filtered_unique_count",
                "filters": filters,
                "value_column": column,
            }

        return {
            "operation":
                "calculate_unique_count",
            "column": column,
        }

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "how many",
            "count",
            "number of",
            "record count",
            "number of rows",
            "rows",
        )
    ):

        column = _count_column(
            question,
            df,
        )

        if filtered:

            return {
                "operation":
                    "filtered_count",
                "filters": filters,
                "count_column": column,
            }

        return {
            "operation":
                "calculate_count",
            "column": column,
        }

    # --------------------------------------------------------
    # AVERAGE
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "average",
            "avg",
            "mean",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if not value:
            raise ValueError(
                "Could not determine "
                "the average column."
            )

        if filtered:

            return {
                "operation":
                    "filtered_average",
                "filters": filters,
                "value_column": value,
            }

        if grouping_intent:

            group = _group_column(
                question,
                df,
            )

            if group:

                return {
                    "operation":
                        "group_and_average",
                    "group_column": group,
                    "value_column": value,
                }

        return {
            "operation":
                "calculate_average",
            "column": value,
        }

    # --------------------------------------------------------
    # MINIMUM
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "minimum",
            "minimum value",
            "lowest value",
            "smallest value",
            "lowest",
            "least",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if value:

            return {
                "operation": (
                    "filtered_min"
                    if filtered
                    else "calculate_min"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                **(
                    {
                        "value_column": value
                    }
                    if filtered
                    else {
                        "column": value
                    }
                ),
            }

    # --------------------------------------------------------
    # MAXIMUM
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "maximum value",
            "highest value",
            "largest value",
            "maximum",
            "highest",
            "largest",
            "greatest",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if value:

            return {
                "operation": (
                    "filtered_max"
                    if filtered
                    else "calculate_max"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                **(
                    {
                        "value_column": value
                    }
                    if filtered
                    else {
                        "column": value
                    }
                ),
            }

    # --------------------------------------------------------
    # VALUE FREQUENCY
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "frequency",
            "frequencies",
            "distribution",
            "most common",
            "value counts",
            "how often",
        )
    ):

        column = _find_column(
            question,
            get_dataset_columns(df),
        )

        column = (
            column
            or _group_column(
                question,
                df,
            )
        )

        if column:

            return {
                "operation": (
                    "filtered_value_counts"
                    if filtered
                    else "value_counts"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                "column": column,
            }

    # --------------------------------------------------------
    # GROUPED OPERATIONS
    # --------------------------------------------------------

    if grouping_intent or any(
        phrase in q
        for phrase in (
            "by city",
            "by product",
            "by category",
            "by department",
            "by region",
            "by state",
            "by country",
            "group by",
            "grouped",
        )
    ):

        group = _group_column(
            question,
            df,
        )

        value = _value_column(
            question,
            df,
        )

        if group and value:

            if (
                "average" in q
                or "mean" in q
            ):

                return {
                    "operation": (
                        "filtered_group_and_average"
                        if filtered
                        else "group_and_average"
                    ),
                    **(
                        {
                            "filters": filters
                        }
                        if filtered
                        else {}
                    ),
                    "group_column": group,
                    "value_column": value,
                }

            return {
                "operation": (
                    "filtered_group_and_sum"
                    if filtered
                    else "group_and_sum"
                ),
                **(
                    {
                        "filters": filters
                    }
                    if filtered
                    else {}
                ),
                "group_column": group,
                "value_column": value,
            }

    # --------------------------------------------------------
    # SUM / TOTAL
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "total",
            "sum",
            "revenue",
            "sales",
            "total revenue",
            "total sales",
            "total amount",
            "total cost",
            "total profit",
        )
    ):

        value = _value_column(
            question,
            df,
        )

        if not value:
            raise ValueError(
                "Could not determine "
                "the numeric column."
            )

        if filtered:

            return {
                "operation":
                    "filtered_sum",
                "filters": filters,
                "value_column": value,
            }

        return {
            "operation":
                "calculate_sum",
            "column": value,
        }

    raise ValueError(
        "Could not determine an analysis "
        "operation from the question."
    )


# ============================================================
# OPTIONAL GEMINI PLANNER
# ============================================================

def _gemini_plan(
    question: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Ask Gemini for a plan only when enabled."""

    if not USE_GEMINI or client is None:
        raise RuntimeError(
            "Gemini fallback is disabled."
        )

    prompt = f"""
Return exactly one JSON analysis plan.

USER QUESTION:
{question}

DATASET PROFILE:
{json.dumps(profile, indent=2, default=str)}

SUPPORTED OPERATIONS:
{json.dumps(
    sorted(SUPPORTED_OPERATIONS),
    indent=2,
)}

RULES:

1. Use only actual dataset columns.

2. Never invent column names.

3. Never invent filter values.

4. Filters must contain:
   column
   operator
   value

5. Equality uses "=".

6. Supported filter operators:
   =
   !=
   >
   >=
   <
   <=
   contains
   between

7. Do not calculate any result.

8. Return JSON only.

9. For top_n and filtered_top_n,
   n must be an integer.

10. For filtered operations,
    all conditions must be inside filters.

11. Return exactly one operation.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return extract_json(
        response.text # pyright: ignore[reportArgumentType]
    )


# ============================================================
# CHOOSE ANALYSIS
# ============================================================

def choose_analysis(
    question: str,
    profile: Optional[
        Dict[str, Any]
    ],
    df: Optional[
        pd.DataFrame
    ] = None,
) -> Dict[str, Any]:
    """
    Deterministic first.

    Gemini is only an optional fallback.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if df is None:
        raise ValueError(
            "choose_analysis requires "
            "the real DataFrame."
        )

    normalized_profile = normalize_profile(
        df,
        profile,
    )

    try:

        plan = deterministic_plan(
            question,
            df,
        )

        return validate_plan(
            plan,
            normalized_profile,
        )

    except Exception as deterministic_error:

        if not USE_GEMINI or client is None:
            raise ValueError(
                "Could not understand the "
                "question locally: "
                f"{deterministic_error}"
            ) from deterministic_error

        try:

            plan = _gemini_plan(
                question,
                normalized_profile,
            )

            return validate_plan(
                plan,
                normalized_profile,
            )

        except Exception as gemini_error:

            raise ValueError(
                "Could not create a valid "
                "analysis plan. "
                f"Local planner: "
                f"{deterministic_error}; "
                f"Gemini fallback: "
                f"{gemini_error}"
            ) from gemini_error


# ============================================================
# EXECUTION
# ============================================================

def execute_analysis(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Any:
    """
    Execute a validated analysis plan.

    Python performs all actual calculations.
    """

    validate_dataframe(df)

    plan = normalize_plan(
        df,
        plan,
    )

    operation = plan[
        "operation"
    ]

    # --------------------------------------------------------
    # BASIC OPERATIONS
    # --------------------------------------------------------

    if operation == "calculate_sum":

        return calculate_sum(
            df,
            plan["column"],
        )

    if operation == "calculate_average":

        return calculate_average(
            df,
            plan["column"],
        )

    if operation == "calculate_count":

        return calculate_count(
            df,
            plan["column"],
        )

    if operation == "calculate_unique_count":

        return calculate_unique_count(
            df,
            plan["column"],
        )

    if operation == "calculate_min":

        return calculate_min(
            df,
            plan["column"],
        )

    if operation == "calculate_max":

        return calculate_max(
            df,
            plan["column"],
        )

    # --------------------------------------------------------
    # GROUP OPERATIONS
    # --------------------------------------------------------

    if operation == "group_and_sum":

        return group_and_sum(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    if operation == "group_and_average":

        return group_and_average(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    if operation == "group_and_count":

        return group_and_count(
            df,
            plan["group_column"],
        )

    # --------------------------------------------------------
    # TOP N
    # --------------------------------------------------------

    if operation == "top_n":

        return top_n(
            df,
            plan["group_column"],
            plan["value_column"],
            plan.get(
                "n",
                5,
            ),
        )

    # --------------------------------------------------------
    # PERCENTAGE
    # --------------------------------------------------------

    if operation == "percentage_of_total":

        return percentage_of_total(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    if operation == "monthly_sum":

        return monthly_sum(
            df,
            plan["date_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # VALUE COUNTS
    # --------------------------------------------------------

    if operation == "value_counts":

        return value_counts(
            df,
            plan["column"],
        )

    # --------------------------------------------------------
    # FILTERED SUM
    # --------------------------------------------------------

    if operation == "filtered_sum":

        return filtered_sum(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED AVERAGE
    # --------------------------------------------------------

    if operation == "filtered_average":

        return filtered_average(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED COUNT
    # --------------------------------------------------------

    if operation == "filtered_count":

        return filtered_count(
            df,
            plan["filters"],
            plan["count_column"],
        )

    # --------------------------------------------------------
    # FILTERED UNIQUE COUNT
    # --------------------------------------------------------

    if operation == "filtered_unique_count":

        return filtered_unique_count(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED MIN
    # --------------------------------------------------------

    if operation == "filtered_min":

        return filtered_min(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED MAX
    # --------------------------------------------------------

    if operation == "filtered_max":

        return filtered_max(
            df,
            plan["filters"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED GROUP SUM
    # --------------------------------------------------------

    if operation == "filtered_group_and_sum":

        return filtered_group_and_sum(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED GROUP AVERAGE
    # --------------------------------------------------------

    if operation == "filtered_group_and_average":

        return filtered_group_and_average(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
        )

    # --------------------------------------------------------
    # FILTERED VALUE COUNTS
    # --------------------------------------------------------

    if operation == "filtered_value_counts":

        return filtered_value_counts(
            df,
            plan["filters"],
            plan["column"],
        )

    # --------------------------------------------------------
    # FILTERED TOP N
    # --------------------------------------------------------

    if operation == "filtered_top_n":

        return filtered_top_n(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
            plan.get(
                "n",
                5,
            ),
        )

    raise ValueError(
        f"Unsupported operation: "
        f"{operation}"
    )


# ============================================================
# LOCAL RESULT EXPLANATION
# ============================================================

def _local_explanation(
    question: str,
    plan: Dict[str, Any],
    result: Any,
) -> str:
    """
    Produce a safe fallback explanation
    without requiring Gemini.
    """

    operation = plan.get(
        "operation",
        "",
    )

    data = serialize_result(
        result
    )

    if operation in {
        "calculate_sum",
        "filtered_sum",
    }:
        value_column = plan.get(
            "column"
        ) or plan.get(
            "value_column"
        )

        if operation == "filtered_sum":
            return (
                f"The total of "
                f"{value_column} "
                f"for the requested "
                f"conditions is "
                f"{data}."
            )

        return (
            f"The total of "
            f"{value_column} is "
            f"{data}."
        )

    if operation in {
        "calculate_average",
        "filtered_average",
    }:
        value_column = plan.get(
            "column"
        ) or plan.get(
            "value_column"
        )

        if operation == "filtered_average":
            return (
                f"The average of "
                f"{value_column} "
                f"for the requested "
                f"conditions is "
                f"{data}."
            )

        return (
            f"The average of "
            f"{value_column} is "
            f"{data}."
        )

    if operation in {
        "calculate_count",
        "filtered_count",
    }:
        count_column = plan.get(
            "column"
        ) or plan.get(
            "count_column"
        )

        if operation == "filtered_count":
            return (
                f"The number of records "
                f"matching the requested "
                f"conditions is {data}."
            )

        return (
            f"The number of records "
            f"counted in {count_column} "
            f"is {data}."
        )

    if operation in {
        "calculate_unique_count",
        "filtered_unique_count",
    }:
        value_column = plan.get(
            "column"
        ) or plan.get(
            "value_column"
        )

        return (
            f"There are {data} "
            f"unique values in "
            f"{value_column}."
        )

    if operation in {
        "calculate_min",
        "filtered_min",
    }:
        value_column = plan.get(
            "column"
        ) or plan.get(
            "value_column"
        )

        return (
            f"The minimum value of "
            f"{value_column} is "
            f"{data}."
        )

    if operation in {
        "calculate_max",
        "filtered_max",
    }:
        value_column = plan.get(
            "column"
        ) or plan.get(
            "value_column"
        )

        return (
            f"The maximum value of "
            f"{value_column} is "
            f"{data}."
        )

    if operation == "monthly_sum":
        return (
            "The monthly result is "
            f"{data}."
        )

    if operation == "percentage_of_total":
        return (
            "The percentage-of-total "
            f"result is {data}."
        )

    if operation in {
        "group_and_sum",
        "group_and_average",
        "group_and_count",
        "top_n",
        "filtered_group_and_sum",
        "filtered_group_and_average",
        "filtered_top_n",
        "value_counts",
        "filtered_value_counts",
    }:
        return (
            "The analysis result is "
            f"{data}."
        )

    return str(data)


# ============================================================
# GEMINI RESULT EXPLANATION
# ============================================================

def explain_result(
    question: str,
    plan: Dict[str, Any],
    result: Any,
) -> str:
    """
    Explain the actual Python result.

    Gemini is optional. When disabled,
    a deterministic explanation is returned.
    """

    result_data = serialize_result(
        result
    )

    if not USE_GEMINI or client is None:
        return _local_explanation(
            question,
            plan,
            result_data,
        )

    prompt = f"""
You are an expert data analyst.

Answer the user's question using ONLY
the ACTUAL PYTHON RESULT.

USER QUESTION:
{question}

ANALYSIS PLAN:
{json.dumps(
    plan,
    indent=2,
    default=str,
)}

PYTHON RESULT:
{json.dumps(
    result_data,
    indent=2,
    default=str,
)}

RULES:

1. Never invent numbers.

2. Use only the actual Python result.

3. Be concise but useful.

4. Format large numbers with commas.

5. Explain the key finding.

6. If the result is grouped data,
   identify the highest relevant group
   when that is directly evident.

7. If the result is a percentage table,
   identify the largest contribution
   when directly evident.

8. If the result is monthly data,
   identify the highest month
   when directly evident.

9. For unique-count operations,
   clearly say unique or distinct.

10. For filtered operations,
    mention the applied conditions.

11. Do not confuse record count
    with unique count.

12. Do not mention internal prompts.

13. Do not mention Gemini.

14. Do not perform a new calculation.

15. Return only the final natural-language answer.
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        if (
            response is not None
            and getattr(
                response,
                "text",
                None,
            )
        ):
            return response.text.strip() # pyright: ignore[reportOptionalMemberAccess]

    except Exception:
        # Gemini explanation is optional.
        # Never let an explanation failure
        # destroy a successful Python analysis.
        pass

    return _local_explanation(
        question,
        plan,
        result_data,
    )


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    profile: Optional[
        Dict[str, Any]
    ],
    question: str,
) -> Dict[str, Any]:
    """
    Complete AI Data Analyst pipeline.

    Flow:

        User Question
              |
              v
        Deterministic Planner
              |
              v
        Optional Gemini fallback
              |
              v
        Normalize Plan
              |
              v
        Validate Columns
              |
              v
        Python Executes
              |
              v
        Actual Result
              |
              v
        Optional Explanation
              |
              v
        Final Response

    Important:

    Python performs the actual calculation.
    Gemini never supplies the numeric result.
    """

    validate_dataframe(df)

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    normalized_profile = normalize_profile(
        df,
        profile,
    )

    # --------------------------------------------------------
    # STEP 1
    # Choose operation.
    # --------------------------------------------------------

    plan = choose_analysis(
        question,
        normalized_profile,
        df=df,
    )

    # --------------------------------------------------------
    # STEP 2
    # Normalize and validate again.
    #
    # This protects the execution boundary.
    # --------------------------------------------------------

    plan = normalize_plan(
        df,
        plan,
    )

    # --------------------------------------------------------
    # STEP 3
    # Execute actual Python calculation.
    # --------------------------------------------------------

    result = execute_analysis(
        df,
        plan,
    )

    # --------------------------------------------------------
    # STEP 4
    # Explain actual result.
    # --------------------------------------------------------

    explanation = explain_result(
        question,
        plan,
        result,
    )

    # --------------------------------------------------------
    # STEP 5
    # Return structured response.
    # --------------------------------------------------------

    return {
        "plan": plan,
        "result": serialize_result(
            result
        ),
        "explanation": explanation,
    }


# ============================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================

def execute_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Any:
    """
    Backward-compatible alias.

    Older application code may call execute_plan().
    """

    return execute_analysis(
        df,
        plan,
    )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    test_df = pd.DataFrame(
        {
            "City": [
                "Delhi",
                "Delhi",
                "Mumbai",
            ],
            "Product": [
                "Laptop",
                "Phone",
                "Laptop",
            ],
            "Revenue": [
                100,
                50,
                200,
            ],
        }
    )

    test_profile = build_profile(
        test_df
    )

    test_questions = [
        "What is the total revenue?",
        "What is the total revenue in Delhi?",
        "What is the total revenue in Delhi for Laptop?",
        "What is the average revenue?",
        "How many orders are in Delhi?",
        "How many unique customers are in Delhi?",
        "What is the revenue by city?",
        "What is the highest revenue?",
        "What is the highest revenue by city?",
        "What are the top 2 cities by revenue?",
        "What is the frequency of products?",
    ]

    for test_question in test_questions:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "QUESTION:"
        )

        print(
            test_question
        )

        try:

            output = run_analysis(
                test_df,
                test_profile,
                test_question,
            )

            print(
                "\nPLAN:"
            )

            print(
                json.dumps(
                    output["plan"],
                    indent=2,
                    default=str,
                )
            )

            print(
                "\nRESULT:"
            )

            print(
                json.dumps(
                    output["result"],
                    indent=2,
                    default=str,
                )
            )

            print(
                "\nEXPLANATION:"
            )

            print(
                output["explanation"]
            )

        except Exception as exc:

            print(
                "\nERROR:"
            )

            print(exc)