from typing import Any, Dict

import pandas as pd

import analysis_tools as tools


def validate_query(
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate query plan dictionary and raise ValueError if invalid."""

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if not isinstance(plan, dict):
        raise ValueError(
            "Query must be a dictionary."
        )

    operation = plan.get("operation")

    if not operation:
        raise ValueError(
            "Query missing 'operation' field."
        )

    op = str(operation).strip().lower()

    # ========================================================
    # SUPPORTED OPERATIONS
    # ========================================================

    col_ops = {
        "calculate_sum",
        "sum",
        "calculate_average",
        "average",
        "mean",
        "calculate_min",
        "min",
        "calculate_max",
        "max",
        "calculate_unique_count",
        "unique_count",
        "value_counts",
        "filtered_value_counts",
    }

    group_ops = {
        "group_and_sum",
        "group_sum",
        "filtered_group_sum",
        "group_and_average",
        "group_average",
        "filtered_group_average",
        "group_and_count",
        "group_count",
        "top_n",
        "filtered_top_n",
        "percentage_of_total",
        "filtered_percentage_of_total",
    }

    other_ops = {
        "calculate_count",
        "count",
        "monthly_sum",
        "filtered_monthly_sum",
        "monthly_average",
        "filtered_monthly_average",
        "monthly_count",
        "filtered_monthly_count",
    }

    valid_ops = col_ops | group_ops | other_ops

    # ========================================================
    # OPERATION VALIDATION
    # ========================================================

    if op not in valid_ops:
        raise ValueError(
            f"Unsupported query operation: '{operation}'"
        )

    # ========================================================
    # COLUMN VALIDATION
    # ========================================================

    col = (
        plan.get("column")
        or plan.get("value_column")
        or plan.get("count_column")
    )

    if op in col_ops and not col:
        raise ValueError(
            f"Operation '{operation}' is missing 'column'."
        )

    # ========================================================
    # GROUP VALIDATION
    # ========================================================

    group_col = (
        plan.get("group_by")
        or plan.get("group_column")
    )

    if op in group_ops and not group_col:
        raise ValueError(
            f"Operation '{operation}' is missing 'group_by'."
        )

    # ========================================================
    # TOP N VALIDATION
    # ========================================================

    if op in {
        "top_n",
        "filtered_top_n",
    } and "n" not in plan:
        raise ValueError(
            f"Operation '{operation}' is missing 'n'."
        )

    # ========================================================
    # NORMALIZE PLAN
    # ========================================================

    plan["operation"] = op

    # Preserve both public/internal group names.
    if group_col:
        plan["group_by"] = group_col
        plan["group_column"] = group_col

    # Preserve the canonical column name.
    if col:
        plan["column"] = col

    return plan


def execute_query(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Any:
    # sourcery skip: low-code-quality
    """Execute an analysis plan against a pandas DataFrame."""

    validated_plan = validate_query(plan)

    operation = validated_plan["operation"]

    col = (
        validated_plan.get("column")
        or validated_plan.get("value_column")
        or validated_plan.get("count_column")
    )

    group_col = (
        validated_plan.get("group_by")
        or validated_plan.get("group_column")
    )

    date_col = (
        validated_plan.get("date_column")
        or validated_plan.get("date")
        or "Date"
    )

    filters = validated_plan.get(
        "filters",
        [],
    )

    n = validated_plan.get(
        "n",
        5,
    )

    has_filters = bool(filters)

    # ========================================================
    # BASIC AGGREGATIONS
    # ========================================================

    match operation:

        case "calculate_sum" | "sum":
            if has_filters:
                return tools.filtered_sum(
                    df,
                    filters,
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.calculate_sum(
                df,
                col, # pyright: ignore[reportArgumentType]
            )

        case "calculate_average" | "average" | "mean":
            if has_filters:
                return tools.filtered_average(
                    df,
                    filters,
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.calculate_average(
                df,
                col, # pyright: ignore[reportArgumentType]
            )

        case "calculate_count" | "count":
            if has_filters:
                return tools.filtered_count(
                    df,
                    filters,
                    col,
                )

            return tools.calculate_count(
                df,
                col,
            )

        case "calculate_unique_count" | "unique_count":
            if has_filters:
                return tools.filtered_unique_count(
                    df,
                    filters,
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.calculate_unique_count(
                df,
                col, # pyright: ignore[reportArgumentType]
            )

        case "calculate_min" | "min":
            if has_filters:
                return tools.filtered_min(
                    df,
                    filters,
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.calculate_min(
                df,
                col, # pyright: ignore[reportArgumentType]
            )

        case "calculate_max" | "max":
            if has_filters:
                return tools.filtered_max(
                    df,
                    filters,
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.calculate_max(
                df,
                col, # pyright: ignore[reportArgumentType]
            )

        # ====================================================
        # GROUP OPERATIONS
        # ====================================================

        case (
            "group_and_sum"
            | "group_sum"
            | "filtered_group_sum"
        ):
            if has_filters:
                return tools.filtered_group_and_sum(
                    df,
                    filters,
                    group_col, # pyright: ignore[reportArgumentType]
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.group_and_sum(
                df,
                group_col, # pyright: ignore[reportArgumentType]
                col, # pyright: ignore[reportArgumentType]
            )

        case (
            "group_and_average"
            | "group_average"
            | "filtered_group_average"
        ):
            if has_filters:
                return tools.filtered_group_and_average(
                    df,
                    filters,
                    group_col, # pyright: ignore[reportArgumentType]
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.group_and_average(
                df,
                group_col, # pyright: ignore[reportArgumentType]
                col, # pyright: ignore[reportArgumentType]
            )

        case "group_and_count" | "group_count":
            if has_filters:
                filtered_df = tools.apply_filters(
                    df,
                    filters,
                )

                return tools.group_and_count(
                    filtered_df,
                    group_col, # pyright: ignore[reportArgumentType]
                )

            return tools.group_and_count(
                df,
                group_col, # pyright: ignore[reportArgumentType]
            )

        # ====================================================
        # VALUE COUNTS
        # ====================================================

        case "value_counts" | "filtered_value_counts":
            if has_filters:
                return tools.filtered_value_counts(
                    df,
                    filters,
                    col, # pyright: ignore[reportArgumentType]
                )

            return tools.value_counts(
                df,
                col, # pyright: ignore[reportArgumentType]
            )

        # ====================================================
        # TOP N
        # ====================================================

        case "top_n" | "filtered_top_n":
            if has_filters:
                return tools.filtered_top_n(
                    df,
                    filters,
                    group_col, # pyright: ignore[reportArgumentType]
                    col, # pyright: ignore[reportArgumentType]
                    n,
                )

            return tools.top_n(
                df,
                group_col, # pyright: ignore[reportArgumentType]
                col, # pyright: ignore[reportArgumentType]
                n,
            )

        # ====================================================
        # PERCENTAGE OF TOTAL
        # ====================================================

        case (
            "percentage_of_total"
            | "filtered_percentage_of_total"
        ):
            if has_filters:
                return tools.filtered_percentage_of_total( # pyright: ignore[reportAttributeAccessIssue]
                    df,
                    filters,
                    group_col,
                    col,
                )

            return tools.percentage_of_total(
                df,
                group_col, # pyright: ignore[reportArgumentType]
                col, # pyright: ignore[reportArgumentType]
            )

        # ====================================================
        # MONTHLY ANALYSIS
        # ====================================================

        case "monthly_sum" | "filtered_monthly_sum":
            if has_filters:
                return tools.filtered_monthly_sum( # pyright: ignore[reportAttributeAccessIssue]
                    df,
                    filters,
                    date_col,
                    col,
                )

            return tools.monthly_sum(
                df,
                date_col,
                col, # type: ignore
            )

        case (
            "monthly_average"
            | "filtered_monthly_average"
        ):
            if has_filters:
                return tools.filtered_monthly_average( # pyright: ignore[reportAttributeAccessIssue]
                    df,
                    filters,
                    date_col,
                    col,
                )

            return tools.monthly_average( # pyright: ignore[reportAttributeAccessIssue]
                df,
                date_col,
                col,
            )

        case (
            "monthly_count"
            | "filtered_monthly_count"
        ):
            if has_filters:
                return tools.filtered_monthly_count( # pyright: ignore[reportAttributeAccessIssue]
                    df,
                    filters,
                    date_col,
                )

            return tools.monthly_count( # pyright: ignore[reportAttributeAccessIssue]
                df,
                date_col,
            )

        # ====================================================
        # SAFETY FALLBACK
        # ====================================================

        case _:
            raise ValueError(
                f"Unsupported query operation: '{operation}'"
            )