
from typing import Any, Dict

import pandas as pd

from analysis_tools import (
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
    calculate_average,
    calculate_count,
    calculate_max,
    calculate_min,
    calculate_sum,
    calculate_unique_count,
    validate_dataframe,
)


# ============================================================
# SUPPORTED OPERATIONS
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


# ============================================================
# QUERY VALIDATION
# ============================================================

def validate_query(
    query: Any,
) -> Dict[str, Any]:
    """
    Validate and normalize a query definition.

    Expected query structure:

        {
            "operation": "sum",
            "column": "Sales",
            "filters": [...]
        }

    Grouped operations additionally require:

        "group_by": "Region"

    top_n additionally requires:

        "n": 5
    """

    if not isinstance(query, dict):
        raise ValueError(
            "Query must be a dictionary."
        )

    if "operation" not in query:
        raise ValueError(
            "Query is missing 'operation'."
        )

    operation = str(
        query["operation"]
    ).strip().lower()

    if not operation:
        raise ValueError(
            "Query operation cannot be empty."
        )

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported query operation "
            f"'{operation}'."
        )

    result = dict(query)
    result["operation"] = operation

    operations_requiring_column = {
        "sum",
        "average",
        "unique_count",
        "min",
        "max",
        "group_sum",
        "group_average",
        "top_n",
    }

    if operation in operations_requiring_column:
        if "column" not in query:
            raise ValueError(
                "Query is missing 'column'."
            )

        if not query["column"]:
            raise ValueError(
                "Query has an empty 'column'."
            )

    operations_requiring_group_by = {
        "group_sum",
        "group_average",
        "group_count",
        "top_n",
    }

    if operation in operations_requiring_group_by:
        if "group_by" not in query:
            raise ValueError(
                "Query is missing 'group_by'."
            )

        if not query["group_by"]:
            raise ValueError(
                "Query has an empty 'group_by'."
            )

    if operation == "top_n":
        if "n" not in query:
            raise ValueError(
                "Query is missing 'n'."
            )

        try:
            n = int(query["n"])
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Query 'n' must be an integer."
            ) from exc

        if n <= 0:
            raise ValueError(
                "Query 'n' must be greater than zero."
            )

        result["n"] = n

    if "filters" in query:
        filters = query["filters"]

        if not isinstance(filters, list):
            raise ValueError(
                "Query 'filters' must be a list."
            )

        result["filters"] = filters

    return result


# ============================================================
# QUERY EXECUTION
# ============================================================

def execute_query(
    df: pd.DataFrame,
    query: Dict[str, Any],
) -> Any:    # sourcery skip: low-code-quality
    """
    Execute a validated analysis query against a DataFrame.

    The query engine acts as a dispatcher between the query
    definition and the functions implemented in analysis_tools.
    """

    validate_dataframe(df)

    normalized_query = validate_query(
        query
    )

    operation = normalized_query[
        "operation"
    ]

    column = normalized_query.get(
        "column"
    )

    group_by = normalized_query.get(
        "group_by"
    )

    n = normalized_query.get(
        "n"
    )

    filters = normalized_query.get(
        "filters"
    )

    # --------------------------------------------------------
    # VALIDATE COLUMNS
    # --------------------------------------------------------

    if (
        column is not None
        and column not in df.columns
    ):
        raise ValueError(
            f"Column '{column}' "
            "does not exist in the DataFrame."
        )

    if (
        group_by is not None
        and group_by not in df.columns
    ):
        raise ValueError(
            f"Column '{group_by}' "
            "does not exist in the DataFrame."
        )

    # --------------------------------------------------------
    # NO FILTERS
    # --------------------------------------------------------

    if not filters:

        if operation == "sum":
            return calculate_sum(
                df,
                column, # pyright: ignore[reportArgumentType]
            )

        if operation == "average":
            return calculate_average(
                df,
                column, # pyright: ignore[reportArgumentType]
            )

        if operation == "count":
            return (
                calculate_count(df)
                if column is None
                else calculate_count(
                    df,
                    column,
                )
            )
        elif operation == "group_average":
            return group_and_average(
                df,
                group_by, # pyright: ignore[reportArgumentType]
                column, # pyright: ignore[reportArgumentType]
            )

        elif operation == "group_count":
            return group_and_count(
                df,
                group_by, # pyright: ignore[reportArgumentType]
            )

        elif operation == "group_sum":
            return group_and_sum(
                df,
                group_by, # pyright: ignore[reportArgumentType]
                column, # pyright: ignore[reportArgumentType]
            )

        elif operation == "max":
            return calculate_max(
                df,
                column, # pyright: ignore[reportArgumentType]
            )

        elif operation == "min":
            return calculate_min(
                df,
                column, # pyright: ignore[reportArgumentType]
            )

        elif operation == "top_n":
            return top_n(
                df,
                group_by, # pyright: ignore[reportArgumentType]
                column, # pyright: ignore[reportArgumentType]
                n, # pyright: ignore[reportArgumentType]
            )

        elif operation == "unique_count":
            return calculate_unique_count(
                df,
                column, # pyright: ignore[reportArgumentType]
            )

        elif operation == "value_counts":
            return value_counts(
                df,
                column, # pyright: ignore[reportArgumentType]
            )

    # --------------------------------------------------------
    # FILTERED OPERATIONS
    # --------------------------------------------------------

    normalized_filters = normalize_filters(
        filters, # pyright: ignore[reportArgumentType]
        df, # pyright: ignore[reportArgumentType]
    )

    if not normalized_filters:
        raise ValueError(
            "No valid filters were provided."
        )

    if operation == "sum":
        return filtered_sum(
            df,
            normalized_filters,
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "average":
        return filtered_average(
            df,
            normalized_filters,
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "count":
        return filtered_count(
            df,
            normalized_filters,
        )

    if operation == "unique_count":
        return filtered_unique_count(
            df,
            normalized_filters,
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "min":
        return filtered_min(
            df,
            normalized_filters,
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "max":
        return filtered_max(
            df,
            normalized_filters,
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "group_sum":
        return filtered_group_and_sum(
            df,
            normalized_filters,
            group_by, # pyright: ignore[reportArgumentType]
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "group_average":
        return filtered_group_and_average(
            df,
            normalized_filters,
            group_by, # pyright: ignore[reportArgumentType]
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "value_counts":
        return filtered_value_counts(
            df,
            normalized_filters,
            column, # pyright: ignore[reportArgumentType]
        )

    if operation == "top_n":
        return filtered_top_n(
            df,
            normalized_filters,
            group_by, # pyright: ignore[reportArgumentType]
            column, # pyright: ignore[reportArgumentType]
            n, # pyright: ignore[reportArgumentType]
        )

    # This should never be reached because
    # validate_query() validates the operation.
    raise ValueError(
        f"Unsupported query operation "
        f"'{operation}'."
    )

