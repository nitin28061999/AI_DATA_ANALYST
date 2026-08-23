# analyst.py

from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis_tools import (
    calculate_sum, # pyright: ignore[reportAttributeAccessIssue]
    calculate_average, # pyright: ignore[reportAttributeAccessIssue]
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
# OFF by default. Normal supported questions make ZERO API calls.

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

USE_GEMINI = os.getenv("ANALYST_USE_GEMINI", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

client = None

if USE_GEMINI:
    try:
        from google import genai

        key = os.getenv("GEMINI_API_KEY")

        if key:
            client = genai.Client(api_key=key)
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


# ============================================================
# GENERAL HELPERS
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that df is a usable pandas DataFrame."""
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")


def get_dataset_columns(df: pd.DataFrame) -> List[str]:
    """Return dataset column names as strings."""
    validate_dataframe(df)
    return [str(c) for c in df.columns]


def _norm(value: Any) -> str:
    """Normalize text for matching."""
    return re.sub(r"\s+", " ", str(value).strip().lower()).strip()


def _safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): _safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return [
            _safe(x)
            for x in value.to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        return {
            str(k): _safe(v)
            for k, v in value.to_dict().items()
        }

    if hasattr(value, "item"):
        with contextlib.suppress(Exception):
            return _safe(value.item())

    if hasattr(value, "isoformat"):
        with contextlib.suppress(Exception):
            return value.isoformat()

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def serialize_result(result: Any) -> Any:
    """Public serialization helper."""
    return _safe(result)


def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response."""
    if not text:
        raise ValueError("Empty JSON response.")

    text = re.sub(
        r"```json|```",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    with contextlib.suppress(json.JSONDecodeError):
        value = json.loads(text)

        if isinstance(value, dict):
            return value

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("No JSON object found.")

    value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError(
            "Analysis plan must be a JSON object."
        )

    return value


# ============================================================
# PROFILE
# ============================================================

def build_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a dataset profile from the real DataFrame."""
    validate_dataframe(df)

    columns = get_dataset_columns(df)

    numeric = [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    dates = [
        c
        for c in columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    return {
        "columns": columns,
        "row_count": len(df),
        "numeric_columns": numeric,
        "datetime_columns": dates,
        "text_columns": [
            c
            for c in columns
            if c not in numeric and c not in dates
        ],
    }


def normalize_profile(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize a supplied profile while treating df as authoritative."""
    validate_dataframe(df)

    generated = build_profile(df)

    if profile is None:
        return generated

    if not isinstance(profile, dict):
        raise ValueError(
            "Dataset profile must be a dictionary."
        )

    result = dict(profile)

    # The actual DataFrame is authoritative.
    result["columns"] = generated["columns"]

    for key, value in generated.items():
        result.setdefault(key, value)

    return result


def _profile_columns(
    profile: Optional[Dict[str, Any]],
) -> List[str]:
    if profile is None:
        return []

    if not isinstance(profile, dict):
        raise ValueError(
            "Dataset profile must be a dictionary."
        )

    columns = profile.get("columns", [])

    if isinstance(columns, dict):
        columns = list(columns.keys())

    return [str(x) for x in columns] if isinstance(columns, (list, tuple)) else []


# ============================================================
# COLUMN SELECTION
# ============================================================

def _find_column(
    question: str,
    columns: List[str],
) -> Optional[str]:
    """Find an explicitly mentioned column."""
    q = _norm(question)

    # Longest names first so that e.g. Customer_ID
    # is preferred over Customer.
    return next(
        (
            c
            for c in sorted(columns, key=len, reverse=True)
            if _norm(c) in q
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
        _norm(key)
        for key in keywords
    ]

    for c in columns:
        name = _norm(c)
        score = 0

        for key in normalized_keywords:
            if name == key:
                score += 100
            elif key in name:
                score += 20

        if score:
            scored.append(
                (
                    score,
                    -len(c),
                    c,
                )
            )

    return None if not scored else max(scored)[2]


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def _text_columns(df: pd.DataFrame) -> List[str]:
    numeric = set(_numeric_columns(df))

    return [
        str(c)
        for c in df.columns
        if str(c) not in numeric
    ]


def _value_column(
    question: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """Choose a numeric value column."""
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

    return found or (
        numeric[0]
        if numeric
        else None
    )


def _group_column(
    question: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """Choose a grouping column."""
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
        ],
    )

    return found or (
        text[0]
        if text
        else None
    )


def _count_column(
    question: str,
    df: pd.DataFrame,
) -> str:
    """Choose the column whose rows should be counted."""
    columns = get_dataset_columns(df)
    q = _norm(question)

    if "invoice" in q:
        found = _semantic_column(
            columns,
            [
                "invoice",
                "invoice_id",
                "invoice id",
            ],
        )

        if found:
            return found

    if "order" in q:
        found = _semantic_column(
            columns,
            [
                "order",
                "order_id",
                "order id",
            ],
        )

        if found:
            return found

    if "transaction" in q:
        found = _semantic_column(
            columns,
            [
                "transaction",
                "transaction_id",
                "transaction id",
            ],
        )

        if found:
            return found

    if "customer" in q:
        found = _semantic_column(
            columns,
            [
                "customer",
                "customer_id",
                "customer id",
            ],
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
    """Choose the column whose distinct values should be counted."""
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
            "city",
            [
                "city",
            ],
        ),
    ]

    for trigger, keys in candidates:
        if trigger in q:
            found = _semantic_column(
                columns,
                keys,
            )

            if found:
                return found

    return (
        _find_column(question, columns)
        or columns[0]
    )


# ============================================================
# FILTERS
# ============================================================

def _coerce(
    value: Any,
    series: pd.Series,
) -> Any:
    """Coerce a textual value to the type used by a DataFrame column."""
    if not isinstance(value, str):
        return value

    value = (
        value
        .strip()
        .strip("\"'")
        .rstrip(".,;!?")
        .strip()
    )

    if pd.api.types.is_numeric_dtype(series):
        with contextlib.suppress(ValueError):
            return (
                float(value)
                if "." in value
                else int(value)
            )

    if pd.api.types.is_bool_dtype(series):
        lowered = value.lower()

        if lowered in {"true", "yes"}:
            return True

        if lowered in {"false", "no"}:
            return False

    return value


def _dataset_value(
    column: str,
    value: Any,
    df: pd.DataFrame,
) -> Any:
    """
    Resolve a textual value against an actual dataset value.

    For example:
        "delhi" -> "Delhi"
        "laptop" -> "Laptop"
    """
    if not isinstance(value, str):
        return value

    value = (
        value
        .strip()
        .strip("\"'")
        .rstrip(".,;!?")
        .strip()
    )

    series = df[column]

    mask = (
        series.astype(str)
        .str.strip()
        .str.lower()
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
    preferred_keywords: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Find a dataset column containing an exact value.

    This is used for natural language filters such as:
        in Delhi
        for Laptop
    """
    value_norm = _norm(value)

    columns = get_dataset_columns(df)

    preferred = preferred_keywords or []

    ordered_columns = sorted(
        columns,
        key=lambda c: (
            0
            if any(
                _norm(key) in _norm(c)
                for key in preferred
            )
            else 1,
            len(c),
        ),
    )

    for column in ordered_columns:
        series = df[column]

        values = (
            series.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if value_norm in set(values):
            return column

    return None


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

    Natural-language equality is only accepted when the value
    actually exists in a DataFrame column.
    """
    columns = get_dataset_columns(df)
    q = question.strip()

    filters: List[Dict[str, Any]] = []
    used_columns: set[str] = set()

    def add(
        column: str,
        operator: str,
        value: Any,
    ) -> None:
        if column in used_columns:
            return

        if isinstance(value, str):
            value = _dataset_value(
                column,
                value,
                df,
            )

        normalized_operator = (
            "="
            if operator == "=="
            else operator
        )

        filters.append(
            {
                "column": column,
                "operator": normalized_operator,
                "value": value,
            }
        )

        used_columns.add(column)

    # --------------------------------------------------------
    # Explicit column + between.
    # Example:
    # Salary between 50000 and 70000
    # --------------------------------------------------------

    for column in columns:
        escaped = re.escape(column)

        pattern = (
            rf"\b{escaped}\b"
            rf"\s+(between)\s+"
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

        if match:
            first_value = match[2].strip()
            second_value = match[3].strip()

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
    # Explicit column + normal operators.
    #
    # Examples:
    # City = Delhi
    # Price > 500
    # Product contains Laptop
    # --------------------------------------------------------

    for column in columns:
        if column in used_columns:
            continue

        escaped = re.escape(column)

        pattern = (
            rf"\b{escaped}\b"
            rf"\s*(>=|<=|!=|==|=|>|<|contains|is)"
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

        operator = match[1].lower()
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
    # Explicit numeric comparison.
    #
    # This is intentionally separate so that:
    # Price > 500
    # Salary >= 50000
    # work reliably even if trailing text follows.
    # --------------------------------------------------------

    for column in columns:
        if column in used_columns:
            continue

        escaped = re.escape(column)

        pattern = (
            rf"\b{escaped}\b"
            rf"\s*(>=|<=|!=|>|<)"
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
    # Generic "between X and Y".
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

        if value_column and value_column not in used_columns:
            add(
                value_column,
                "between",
                [
                    _coerce(generic_between[1], df[value_column]),
                    _coerce(generic_between[2], df[value_column]),
                ],
            )

    # --------------------------------------------------------
    # Natural-language equality:
    #
    # "in Delhi"
    # "for Laptop"
    # "from Delhi"
    #
    # The candidate must actually exist in the dataset.
    # --------------------------------------------------------

    natural_pattern = re.compile(
        r"\b(in|for|from)\s+"
        r"['\"]?([^,;?.]+?)['\"]?"
        r"(?=\s+(?:for|in|from|and|where)\b|[,;?.]|$)",
        re.IGNORECASE,
    )

    natural_matches = natural_pattern.findall(q)

    for keyword, raw_value in natural_matches:
        value = (
            raw_value
            .strip()
            .strip("\"'")
            .strip()
        )

        if not value:
            continue

        preferred_keywords: List[str]

        if keyword.lower() in {"in", "from"}:
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
    # Natural equality using:
    #
    # "City Delhi"
    # "Product Laptop"
    #
    # Only consider explicit dataset column names.
    # --------------------------------------------------------

    for column in columns:
        if column in used_columns:
            continue

        escaped = re.escape(column)

        pattern = (
            rf"\b{escaped}\b"
            rf"\s+"
            rf"['\"]?([^,;?.]+?)['\"]?"
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

        # Avoid interpreting operation words as values.
        if _norm(candidate) in {
            "revenue",
            "sales",
            "amount",
            "profit",
            "average",
            "mean",
            "total",
            "sum",
            "count",
            "maximum",
            "minimum",
            "highest",
            "lowest",
        }:
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


def normalize_filters(
    filters: Any,
    df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Validate and normalize filter definitions."""
    if not isinstance(filters, list):
        raise ValueError(
            "Filters must be a list."
        )

    if not filters:
        raise ValueError(
            "At least one filter is required."
        )

    supported = {
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

    result: List[Dict[str, Any]] = []

    for i, item in enumerate(filters):
        if not isinstance(item, dict):
            raise ValueError(
                f"Filter #{i + 1} must be a dictionary."
            )

        if "column" not in item:
            raise ValueError(
                f"Filter #{i + 1} is missing 'column'."
            )

        if "value" not in item:
            raise ValueError(
                f"Filter #{i + 1} is missing 'value'."
            )

        column = item["column"]

        if not column:
            raise ValueError(
                f"Filter #{i + 1} has an empty column."
            )

        operator = str(
            item.get("operator", "=") or "="
        ).lower().strip()

        if operator == "==":
            operator = "="

        if operator not in supported:
            raise ValueError(
                f"Unsupported filter operator "
                f"'{operator}'."
            )

        value = item["value"]

        if operator == "between":
            if not isinstance(
                value,
                (list, tuple),
            ) or len(value) != 2:
                raise ValueError(
                    "The 'between' filter requires "
                    "exactly two values."
                )

            value = [
                value[0],
                value[1],
            ]

        if df is not None and column in df.columns:
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
                value = str(value).strip()
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
    """Validate that all columns referenced by a plan exist."""
    columns = set(
        get_dataset_columns(df)
    )

    operation = plan.get("operation")

    def require(key: str) -> None:
        value = plan.get(key)

        if not value:
            raise ValueError(
                f"{operation} requires '{key}'."
            )

        if value not in columns:
            raise ValueError(
                f"Column '{value}' does not exist. "
                f"Available columns: {sorted(columns)}"
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
        filters = plan.get("filters")

        if not isinstance(filters, list) or not filters:
            raise ValueError(
                f"{operation} requires 'filters'."
            )

        for i, filter_item in enumerate(filters):
            if not isinstance(filter_item, dict):
                raise ValueError(
                    f"Filter #{i + 1} must be a dictionary."
                )

            if not filter_item.get("column"):
                raise ValueError(
                    f"Filter #{i + 1} is missing 'column'."
                )

            if filter_item["column"] not in columns:
                raise ValueError(
                    f"Filter column "
                    f"'{filter_item['column']}' "
                    f"does not exist."
                )

            if "operator" not in filter_item:
                filter_item["operator"] = "="

            if "value" not in filter_item:
                raise ValueError(
                    f"Filter #{i + 1} is missing 'value'."
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
    """Validate an analysis plan against the dataset profile."""
    if not isinstance(plan, dict):
        raise ValueError(
            "Analysis plan must be a dictionary."
        )

    operation = str(
        plan.get("operation", "")
    ).strip()

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation "
            f"'{operation}'."
        )

    result = dict(plan)
    result["operation"] = operation

    if operation.startswith("filtered_"):
        result["filters"] = normalize_filters(
            result.get("filters")
        )

    if operation in {
        "top_n",
        "filtered_top_n",
    }:
        try:
            n = int(
                result.get("n", 5)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "n must be an integer."
            ) from exc

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        result["n"] = n

    columns = _profile_columns(profile)

    if columns:
        for key in (
            "column",
            "group_column",
            "value_column",
            "count_column",
            "date_column",
        ):
            if (
                key in result
                and result[key]
                and result[key] not in columns
            ):
                raise ValueError(
                    f"Column '{result[key]}' "
                    f"does not exist in the dataset."
                )

        for filter_item in result.get(
            "filters",
            [],
        ):
            if filter_item["column"] not in columns:
                raise ValueError(
                    f"Filter column "
                    f"'{filter_item['column']}' "
                    f"does not exist."
                )

    return result


def normalize_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and normalize a plan against the real DataFrame."""
    validate_dataframe(df)

    if not isinstance(plan, dict):
        raise ValueError(
            "Analysis plan must be a dictionary."
        )

    operation = str(
        plan.get("operation", "")
    ).strip()

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation "
            f"'{operation}'."
        )

    result = dict(plan)
    result["operation"] = operation

    if operation.startswith("filtered_"):
        result["filters"] = normalize_filters(
            result.get("filters"),
            df=df,
        )

    if operation in {
        "top_n",
        "filtered_top_n",
    }:
        try:
            n = int(
                result.get("n", 5)
            )
        except (TypeError, ValueError) as exc:
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
# DETERMINISTIC PLANNER
# ============================================================

def deterministic_plan(
    question: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:  # sourcery skip: low-code-quality
    """Create an analysis plan without using an external API."""
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

    # --------------------------------------------------------
    # Highest / top / best.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
            "highest",
            "largest",
            "greatest",
            "best performing",
            "top ",
            "maximum by",
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
            return {
                "operation": (
                    "filtered_top_n"
                    if filtered
                    else "top_n"
                ),
                **(
                    {"filters": filters}
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
    # Percentage / share.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
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

        if group and value and not filtered:
            return {
                "operation": "percentage_of_total",
                "group_column": group,
                "value_column": value,
            }

    # --------------------------------------------------------
    # Monthly.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
            "monthly",
            "by month",
            "per month",
            "month wise",
            "month-wise",
            "monthly trend",
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
    # Unique / distinct.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
            "unique",
            "distinct",
            "different ",
            "unique customers",
            "distinct customers",
        )
    ):
        column = _unique_column(
            question,
            df,
        )

        if filtered:
            return {
                "operation": "filtered_unique_count",
                "filters": filters,
                "value_column": column,
            }

        return {
            "operation": "calculate_unique_count",
            "column": column,
        }

    # --------------------------------------------------------
    # Count.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
            "how many",
            "count",
            "number of",
            "record count",
        )
    ):
        column = _count_column(
            question,
            df,
        )

        if filtered:
            return {
                "operation": "filtered_count",
                "filters": filters,
                "count_column": column,
            }

        return {
            "operation": "calculate_count",
            "column": column,
        }

    # --------------------------------------------------------
    # Average.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
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
                "Could not determine the average column."
            )

        if filtered:
            return {
                "operation": "filtered_average",
                "filters": filters,
                "value_column": value,
            }

        return {
            "operation": "calculate_average",
            "column": value,
        }

    # --------------------------------------------------------
    # Minimum.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
            "minimum",
            "minimum value",
            "lowest value",
            "smallest value",
            "lowest",
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
                    {"filters": filters}
                    if filtered
                    else {}
                ),
                **(
                    {"value_column": value}
                    if filtered
                    else {"column": value}
                ),
            }

    # --------------------------------------------------------
    # Maximum.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
            "maximum value",
            "highest value",
            "largest value",
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
                    {"filters": filters}
                    if filtered
                    else {}
                ),
                **(
                    {"value_column": value}
                    if filtered
                    else {"column": value}
                ),
            }

    # --------------------------------------------------------
    # Value frequency.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
            "frequency",
            "frequencies",
            "distribution",
            "most common",
            "value counts",
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
                    {"filters": filters}
                    if filtered
                    else {}
                ),
                "column": column,
            }

    # --------------------------------------------------------
    # Grouped operations.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
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
                        {"filters": filters}
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
                    {"filters": filters}
                    if filtered
                    else {}
                ),
                "group_column": group,
                "value_column": value,
            }

    # --------------------------------------------------------
    # Sum / total.
    # --------------------------------------------------------

    if any(
        x in q
        for x in (
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
                "Could not determine the numeric column."
            )

        if filtered:
            return {
                "operation": "filtered_sum",
                "filters": filters,
                "value_column": value,
            }

        return {
            "operation": "calculate_sum",
            "column": value,
        }

    raise ValueError(
        "Could not determine an analysis operation "
        "from the question."
    )


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

    return max(1, int(match[1])) if match else default
    


# ============================================================
# OPTIONAL GEMINI FALLBACK
# ============================================================

def _gemini_plan(
    question: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Ask Gemini for a plan only when explicitly enabled."""
    if not USE_GEMINI or client is None:
        raise RuntimeError(
            "Gemini fallback is disabled."
        )

    prompt = f"""
Return exactly one JSON analysis plan.

Question:

{question}

Dataset profile:

{json.dumps(profile, indent=2, default=str)}

Supported operations:

{json.dumps(sorted(SUPPORTED_OPERATIONS))}

Rules:

- Use only actual columns.
- Filters must contain column, operator and value.
- Equality uses "=".
- Do not invent values.
- Do not calculate.
- Return JSON only.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return extract_json(
        response.text # pyright: ignore[reportArgumentType]
    )


def choose_analysis(
    question: str,
    profile: Optional[Dict[str, Any]],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Deterministic first, Gemini only as an opt-in fallback.

    The real DataFrame is required because deterministic planning
    uses the actual column names and actual dataset values.
    """
    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if df is None:
        raise ValueError(
            "choose_analysis requires the real DataFrame."
        )

    profile = normalize_profile(
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
            profile,
        )

    except Exception as deterministic_error:
        if not USE_GEMINI or client is None:
            raise ValueError(
                "Could not understand the question locally: "
                f"{deterministic_error}"
            ) from deterministic_error

        try:
            plan = _gemini_plan(
                question,
                profile,
            )

            return validate_plan(
                plan,
                profile,
            )

        except Exception as gemini_error:
            raise ValueError(
                "Could not create a valid analysis plan. "
                f"Local planner: {deterministic_error}; "
                f"Gemini fallback: {gemini_error}"
            ) from gemini_error


# ============================================================
# EXECUTION
# ============================================================

def execute_analysis(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Any:
    """Execute a validated analysis plan."""
    plan = normalize_plan(
        df,
        plan,
    )

    op = plan["operation"]

    if op == "calculate_sum":
        return calculate_sum(
            df,
            plan["column"],
        )

    if op == "calculate_average":
        return calculate_average(
            df,
            plan["column"],
        )

    if op == "calculate_count":
        return calculate_count(
            df,
            plan["column"],
        )

    if op == "calculate_unique_count":
        return calculate_unique_count(
            df,
            plan["column"],
        )

    if op == "calculate_min":
        return calculate_min(
            df,
            plan["column"],
        )

    if op == "calculate_max":
        return calculate_max(
            df,
            plan["column"],
        )

    if op == "group_and_sum":
        return group_and_sum(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    if op == "group_and_average":
        return group_and_average(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    if op == "group_and_count":
        return group_and_count(
            df,
            plan["group_column"],
        )

    if op == "top_n":
        return top_n(
            df,
            plan["group_column"],
            plan["value_column"],
            plan.get("n", 5),
        )

    if op == "percentage_of_total":
        return percentage_of_total(
            df,
            plan["group_column"],
            plan["value_column"],
        )

    if op == "monthly_sum":
        return monthly_sum(
            df,
            plan["date_column"],
            plan["value_column"],
        )

    if op == "value_counts":
        return value_counts(
            df,
            plan["column"],
        )

    if op == "filtered_sum":
        return filtered_sum(
            df,
            plan["filters"],
            plan["value_column"],
        )

    if op == "filtered_average":
        return filtered_average(
            df,
            plan["filters"],
            plan["value_column"],
        )

    if op == "filtered_count":
        return filtered_count(
            df,
            plan["filters"],
            plan["count_column"],
        )

    if op == "filtered_unique_count":
        return filtered_unique_count(
            df,
            plan["filters"],
            plan["value_column"],
        )

    if op == "filtered_min":
        return filtered_min(
            df,
            plan["filters"],
            plan["value_column"],
        )

    if op == "filtered_max":
        return filtered_max(
            df,
            plan["filters"],
            plan["value_column"],
        )

    if op == "filtered_group_and_sum":
        return filtered_group_and_sum(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
        )

    if op == "filtered_group_and_average":
        return filtered_group_and_average(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
        )

    if op == "filtered_value_counts":
        return filtered_value_counts(
            df,
            plan["filters"],
            plan["column"],
        )

    if op == "filtered_top_n":
        return filtered_top_n(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
            plan.get("n", 5),
        )

    raise ValueError(
        f"Unsupported operation: {op}"
    )


# Older project code may call this name.
execute_plan = execute_analysis


# ============================================================
# DETERMINISTIC EXPLANATION
# ============================================================

def _fmt(value: Any) -> str:
    """Format numeric results cleanly."""
    value = serialize_result(value)

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        return f"{int(value):,}" if value.is_integer() else f"{value:,.2f}"
    return str(value)


def _filters_text(
    plan: Dict[str, Any],
) -> str:
    """Convert filters into readable text."""
    parts: List[str] = []

    for filter_item in plan.get(
        "filters",
        [],
    ):
        value = filter_item["value"]

        if filter_item["operator"] == "between":
            value = (
                f"{value[0]} and {value[1]}"
            )

        parts.append(
            f"{filter_item['column']} "
            f"{filter_item['operator']} "
            f"{value}"
        )

    return "; ".join(parts)


def _explain(
    question: str,
    plan: Dict[str, Any],
    result: Any,
) -> str:
    """Generate a deterministic explanation."""
    del question

    op = plan["operation"]
    result = serialize_result(result)

    if op in {
        "calculate_sum",
        "filtered_sum",
    }:
        column = plan.get(
            "column",
            plan.get("value_column"),
        )

        text = (
            f"The total {column} "
            f"is {_fmt(result)}."
        )

        if op == "filtered_sum":
            text += (
                f" Filtered for "
                f"{_filters_text(plan)}."
            )

        return text

    if op in {
        "calculate_average",
        "filtered_average",
    }:
        column = plan.get(
            "column",
            plan.get("value_column"),
        )

        text = (
            f"The average {column} "
            f"is {_fmt(result)}."
        )

        if op == "filtered_average":
            text += (
                f" Filtered for "
                f"{_filters_text(plan)}."
            )

        return text

    if op in {
        "calculate_count",
        "filtered_count",
    }:
        if op == "filtered_count":
            return (
                f"There are {_fmt(result)} records "
                f"matching {_filters_text(plan)}."
            )

        return (
            f"There are {_fmt(result)} records."
        )

    if op in {
        "calculate_unique_count",
        "filtered_unique_count",
    }:
        column = plan.get(
            "column",
            plan.get("value_column"),
        )

        text = (
            f"There are {_fmt(result)} "
            f"unique/distinct values in {column}."
        )

        if op == "filtered_unique_count":
            text += (
                f" Filtered for "
                f"{_filters_text(plan)}."
            )

        return text

    if op == "calculate_min":
        return (
            f"The minimum {plan['column']} "
            f"is {_fmt(result)}."
        )

    if op == "calculate_max":
        return (
            f"The maximum {plan['column']} "
            f"is {_fmt(result)}."
        )

    if op == "filtered_min":
        return (
            f"The minimum "
            f"{plan['value_column']} "
            f"is {_fmt(result)}, "
            f"filtered for "
            f"{_filters_text(plan)}."
        )

    if op == "filtered_max":
        return (
            f"The maximum "
            f"{plan['value_column']} "
            f"is {_fmt(result)}, "
            f"filtered for "
            f"{_filters_text(plan)}."
        )

    if op in {
        "top_n",
        "filtered_top_n",
    }:
        if isinstance(result, list) and result:
            row = result[0]

            if isinstance(row, dict):
                group = row.get(
                    plan["group_column"]
                )

                value = row.get(
                    plan["value_column"]
                )

                text = (
                    f"{group} has the highest "
                    f"{plan['value_column']}, "
                    f"with {_fmt(value)}."
                )

                if op == "filtered_top_n":
                    text += (
                        f" Filtered for "
                        f"{_filters_text(plan)}."
                    )

                return text

        return str(result)

    if op in {
        "group_and_sum",
        "group_and_average",
        "group_and_count",
        "filtered_group_and_sum",
        "filtered_group_and_average",
    }:
        text = (
            "Grouped analysis completed by "
            f"{plan['group_column']}."
        )

        if op.startswith("filtered_"):
            text += (
                f" Filtered for "
                f"{_filters_text(plan)}."
            )

        return text

    if op == "percentage_of_total":
        return (
            "Percentage-of-total analysis completed "
            f"by {plan['group_column']}."
        )

    if op == "monthly_sum":
        return (
            f"Monthly {plan['value_column']} "
            f"analysis completed using "
            f"{plan['date_column']}."
        )

    if op in {
        "value_counts",
        "filtered_value_counts",
    }:
        text = (
            f"Value frequencies for "
            f"{plan['column']} are shown in the result."
        )

        if op == "filtered_value_counts":
            text += (
                f" Filtered for "
                f"{_filters_text(plan)}."
            )

        return text

    return str(result)


def explain_result(
    question: str,
    plan: Dict[str, Any],
    result: Any,
) -> str:
    """
    Explain a result locally.

    NEVER calls Gemini.
    """
    return _explain(
        question,
        plan,
        result,
    )


# ============================================================
# MAIN API
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]],
    question: str,
) -> Dict[str, Any]:
    """
    Main public function.

    Normal request path:

        deterministic planner
            ->
        analysis_tools
            ->
        local explanation

    Therefore the normal path consumes ZERO
    Gemini API requests.
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

    plan = choose_analysis(
        question,
        normalized_profile,
        df=df,
    )

    plan = normalize_plan(
        df,
        plan,
    )

    result = execute_analysis(
        df,
        plan,
    )

    explanation = explain_result(
        question,
        plan,
        result,
    )

    return {
        "plan": plan,
        "result": serialize_result(result),
        "explanation": explanation,
    }


# ============================================================
# SIX-QUESTION REGRESSION TEST
# ============================================================

def run_regression_test() -> None:
    """Run the built-in regression tests."""
    df = pd.DataFrame(
        {
            "City": [
                "Delhi",
                "Delhi",
                "Mumbai",
                "Mumbai",
            ],
            "Product": [
                "Laptop",
                "Phone",
                "Laptop",
                "Phone",
            ],
            "Revenue": [
                100,
                50,
                200,
                300,
            ],
            "Customer_ID": [
                "C1",
                "C2",
                "C1",
                "C3",
            ],
            "Invoice_ID": [
                "I1",
                "I2",
                "I3",
                "I4",
            ],
        }
    )

    tests = [
        (
            "What is the total revenue in Delhi?",
            150.0,
        ),
        (
            "What is the average revenue in Mumbai?",
            250.0,
        ),
        (
            "How many invoices are in Delhi?",
            2,
        ),
        (
            "How many unique customers are in Delhi?",
            2,
        ),
        (
            "Which city has the highest revenue?",
            "Mumbai",
        ),
        (
            "What is the total revenue in Delhi for Laptop?",
            100.0,
        ),
    ]

    print("=" * 70)
    print("LOCAL ANALYST REGRESSION TEST")
    print("Gemini enabled:", USE_GEMINI)
    print("=" * 70)

    for question, expected in tests:
        output = run_analysis(
            df,
            None,
            question,
        )

        result = output["result"]

        if isinstance(expected, str):
            if not isinstance(result, list) or not result:
                raise AssertionError(
                    f"\nQuestion: {question}\n"
                    f"Expected: {expected!r}\n"
                    f"Actual: {result!r}\n"
                    f"Plan: {output['plan']}"
                )

            first_row = result[0]

            if not isinstance(first_row, dict):
                raise AssertionError(
                    f"\nQuestion: {question}\n"
                    f"Expected a row dictionary.\n"
                    f"Actual: {result!r}\n"
                    f"Plan: {output['plan']}"
                )

            actual = first_row.get("City")

        else:
            actual = result

        if actual != expected:
            raise AssertionError(
                f"\nQuestion: {question}\n"
                f"Expected: {expected!r}\n"
                f"Actual: {actual!r}\n"
                f"Plan: {output['plan']}"
            )

        print()
        print("QUESTION:", question)
        print("PLAN:", output["plan"])
        print("RESULT:", result)
        print(
            "EXPLANATION:",
            output["explanation"],
        )

    print()
    print("=" * 70)
    print("ALL SIX TESTS PASSED")
    print("Gemini API requests required: 0")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_regression_test()