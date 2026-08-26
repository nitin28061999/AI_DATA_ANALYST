from typing import Any, Dict, List, Optional, cast
import pandas as pd

from analysis_tools import (
    calculate_average,
    calculate_count,
    calculate_max,
    calculate_min,
    calculate_sum,
    calculate_unique_count,
    filtered_average,
    filtered_count,
    filtered_group_and_average,
    filtered_group_and_sum,
    filtered_max,
    filtered_min,
    filtered_sum,
    filtered_top_n,
    filtered_unique_count,
    filtered_value_counts,
    group_and_average,
    group_and_count,
    group_and_sum,
    normalize_filters,
    top_n,
    value_counts,
    validate_dataframe,
)

# ============================================================
# SUPPORTED OPERATIONS & REQUIREMENTS
# ============================================================

SUPPORTED_OPERATIONS = {
    "sum",
    "average",
    "count",
    "unique_count",
    "min",
    "max",
    "group_sum",
    "group_average",
    "group_count",
    "value_counts",
    "top_n",
}

OPERATIONS_REQUIRING_COLUMN = {
    "sum",
    "average",
    "unique_count",
    "min",
    "max",
    "group_sum",
    "group_average",
    "top_n",
}

OPERATIONS_REQUIRING_GROUP_BY = {
    "group_sum",
    "group_average",
    "group_count",
    "top_n",
}


# ============================================================
# QUERY VALIDATION
# ============================================================

def validate_query(query: Any) -> Dict[str, Any]:
    """
    Validate and normalize a query definition dictionary.
    """
    if not isinstance(query, dict):
        raise ValueError("Query must be a dictionary.")

    if "operation" not in query:
        raise ValueError("Query is missing 'operation'.")

    operation = str(query["operation"]).strip().lower()

    if not operation:
        raise ValueError("Query operation cannot be empty.")

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported query operation '{operation}'.")

    result: Dict[str, Any] = dict(query)
    result["operation"] = operation

    # Validate column requirements
    if operation in OPERATIONS_REQUIRING_COLUMN:
        if "column" not in query or not query["column"]:
            raise ValueError(f"Operation '{operation}' requires a non-empty 'column'.")

    # Validate group_by requirements
    if operation in OPERATIONS_REQUIRING_GROUP_BY:
        if "group_by" not in query or not query["group_by"]:
            raise ValueError(f"Operation '{operation}' requires a non-empty 'group_by'.")

    # Validate top_n parameters
    if operation == "top_n":
        if "n" not in query:
            raise ValueError("Query is missing 'n'.")
        try:
            n = int(query["n"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Query 'n' must be an integer.") from exc

        if n <= 0:
            raise ValueError("Query 'n' must be greater than zero.")

        result["n"] = n

    # Validate filters parameter if provided
    if "filters" in query:
        filters = query["filters"]
        if not isinstance(filters, list):
            raise ValueError("Query 'filters' must be a list.")
        result["filters"] = filters

    return result


# ============================================================
# QUERY EXECUTION ENGINE
# ============================================================

def execute_query(df: pd.DataFrame, query: Dict[str, Any]) -> Any:
    # sourcery skip: low-code-quality
    """
    Execute a validated analysis query against a pandas DataFrame.
    """
    validate_dataframe(df)
    normalized_query = validate_query(query)

    operation: str = normalized_query["operation"]
    column: Optional[str] = normalized_query.get("column")
    group_by: Optional[str] = normalized_query.get("group_by")
    n: Optional[int] = normalized_query.get("n")
    filters: Optional[List[Any]] = normalized_query.get("filters")

    # Check for column existence
    if column is not None and column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"Column '{group_by}' does not exist in the DataFrame.")

    # --------------------------------------------------------
    # UNFILTERED OPERATIONS
    # --------------------------------------------------------
    if not filters:
        if operation == "sum":
            return calculate_sum(df, cast(str, column))
        if operation == "average":
            return calculate_average(df, cast(str, column))
        if operation == "count":
            return calculate_count(df) if column is None else calculate_count(df, column)
        if operation == "unique_count":
            return calculate_unique_count(df, cast(str, column))
        if operation == "min":
            return calculate_min(df, cast(str, column))
        if operation == "max":
            return calculate_max(df, cast(str, column))
        if operation == "group_sum":
            return group_and_sum(df, cast(str, group_by), cast(str, column))
        if operation == "group_average":
            return group_and_average(df, cast(str, group_by), cast(str, column))
        if operation == "group_count":
            return group_and_count(df, cast(str, group_by))
        if operation == "value_counts":
            return value_counts(df, cast(str, column))
        if operation == "top_n":
            return top_n(df, cast(str, group_by), cast(str, column), cast(int, n))

    # --------------------------------------------------------
    # FILTERED OPERATIONS
    # --------------------------------------------------------
    normalized_filters = normalize_filters(filters, df) # pyright: ignore[reportArgumentType]
    if not normalized_filters:
        raise ValueError("No valid filters were provided.")

    if operation == "sum":
        return filtered_sum(df, normalized_filters, cast(str, column))
    if operation == "average":
        return filtered_average(df, normalized_filters, cast(str, column))
    if operation == "count":
        return filtered_count(df, normalized_filters)
    if operation == "unique_count":
        return filtered_unique_count(df, normalized_filters, cast(str, column))
    if operation == "min":
        return filtered_min(df, normalized_filters, cast(str, column))
    if operation == "max":
        return filtered_max(df, normalized_filters, cast(str, column))
    if operation == "group_sum":
        return filtered_group_and_sum(df, normalized_filters, cast(str, group_by), cast(str, column))
    if operation == "group_average":
        return filtered_group_and_average(df, normalized_filters, cast(str, group_by), cast(str, column))
    if operation == "value_counts":
        return filtered_value_counts(df, normalized_filters, cast(str, column))
    if operation == "top_n":
        return filtered_top_n(df, normalized_filters, cast(str, group_by), cast(str, column), cast(int, n))

    raise ValueError(f"Unsupported query operation '{operation}'.")