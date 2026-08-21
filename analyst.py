from ai_agent import (
    choose_analysis,
    explain_result,
)

from analysis_tools import (
    calculate_sum,
    calculate_average,
    calculate_count,
    calculate_unique_count,
    calculate_min,
    calculate_max,
    group_and_sum,
    group_and_average,
    top_n,
    percentage_of_total,
    monthly_sum,
    value_counts,
    group_and_count,
    filtered_sum,
    filtered_average,
    filtered_count,
    filtered_unique_count,
)


def run_analysis(
    df,
    profile,
    question
):  # sourcery skip: low-code-quality
    """
    Complete AI Data Analyst workflow.

    Gemini:
        question -> analysis plan

    Python:
        analysis plan -> actual calculation

    Gemini:
        actual calculation -> explanation
    """

    # ========================================================
    # STEP 1 — AI PLAN
    # ========================================================

    plan = choose_analysis(
        question,
        profile
    )

    if not isinstance(
        plan,
        dict
    ):
        raise ValueError(
            "Gemini returned an invalid analysis plan."
        )

    operation = plan.get(
        "operation"
    )

    if not operation:
        raise ValueError(
            "Gemini did not specify an operation."
        )

    # ========================================================
    # STEP 2 — PYTHON EXECUTION
    # ========================================================

    if operation == "calculate_sum":

        column = plan.get("column")

        if not column:
            raise ValueError(
                "No column specified for sum."
            )

        result = calculate_sum(
            df,
            column
        )

    elif operation == "calculate_average":

        column = plan.get("column")

        if not column:
            raise ValueError(
                "No column specified for average."
            )

        result = calculate_average(
            df,
            column
        )

    elif operation == "calculate_count":

        column = plan.get("column")

        result = calculate_count(
            df,
            column
        )

    elif operation == "calculate_unique_count":

        column = plan.get("column")

        if not column:
            raise ValueError(
                "No column specified for unique count."
            )

        result = calculate_unique_count(
            df,
            column
        )

    elif operation == "calculate_min":

        column = plan.get("column")

        if not column:
            raise ValueError(
                "No column specified for minimum."
            )

        result = calculate_min(
            df,
            column
        )

    elif operation == "calculate_max":

        column = plan.get("column")

        if not column:
            raise ValueError(
                "No column specified for maximum."
            )

        result = calculate_max(
            df,
            column
        )

    elif operation == "group_and_sum":

        group_column = plan.get(
            "group_column"
        )

        value_column = plan.get(
            "value_column"
        )

        if not group_column:
            raise ValueError(
                "No group column specified."
            )

        if not value_column:
            raise ValueError(
                "No value column specified."
            )

        result = group_and_sum(
            df,
            group_column,
            value_column
        )

    elif operation == "group_and_average":

        group_column = plan.get(
            "group_column"
        )

        value_column = plan.get(
            "value_column"
        )

        if not group_column:
            raise ValueError(
                "No group column specified."
            )

        if not value_column:
            raise ValueError(
                "No value column specified."
            )

        result = group_and_average(
            df,
            group_column,
            value_column
        )

    elif operation == "top_n":

        result = _extracted_from_run_analysis_182(plan, df)
    elif operation == "percentage_of_total":

        group_column = plan.get(
            "group_column"
        )

        value_column = plan.get(
            "value_column"
        )

        if not group_column:
            raise ValueError(
                "No group column specified."
            )

        if not value_column:
            raise ValueError(
                "No value column specified."
            )

        result = percentage_of_total(
            df,
            group_column,
            value_column
        )

    elif operation == "monthly_sum":

        date_column = plan.get(
            "date_column"
        )

        value_column = plan.get(
            "value_column"
        )

        if not date_column:
            raise ValueError(
                "No date column specified."
            )

        if not value_column:
            raise ValueError(
                "No value column specified."
            )

        result = monthly_sum(
            df,
            date_column,
            value_column
        )

    elif operation == "value_counts":

        column = plan.get(
            "column"
        )

        if not column:
            raise ValueError(
                "No column specified."
            )

        result = value_counts(
            df,
            column
        )

    elif operation == "group_and_count":

        group_column = plan.get(
            "group_column"
        )

        if not group_column:
            raise ValueError(
                "No group column specified."
            )

        result = group_and_count(
            df,
            group_column
        )

    elif operation == "filtered_sum":

        filters = plan.get(
            "filters"
        )

        value_column = plan.get(
            "value_column"
        )

        if not filters:
            raise ValueError(
                "No filters specified."
            )

        if not value_column:
            raise ValueError(
                "No value column specified."
            )

        result = filtered_sum(
            df,
            filters,
            value_column
        )

    elif operation == "filtered_average":

        filters = plan.get(
            "filters"
        )

        value_column = plan.get(
            "value_column"
        )

        if not filters:
            raise ValueError(
                "No filters specified."
            )

        if not value_column:
            raise ValueError(
                "No value column specified."
            )

        result = filtered_average(
            df,
            filters,
            value_column
        )

    elif operation == "filtered_count":

        filters = plan.get(
            "filters"
        )

        count_column = plan.get(
            "count_column"
        )

        if not filters:
            raise ValueError(
                "No filters specified."
            )

        result = filtered_count(
            df,
            filters,
            count_column
        )

    elif operation == "filtered_unique_count":

        filters = plan.get(
            "filters"
        )

        value_column = plan.get(
            "value_column"
        )

        if not filters:
            raise ValueError(
                "No filters specified."
            )

        if not value_column:
            raise ValueError(
                "No value column specified."
            )

        result = filtered_unique_count(
            df,
            filters,
            value_column
        )

    else:

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # ========================================================
    # STEP 3 — AI EXPLANATION
    # ========================================================

    explanation = explain_result(
        question,
        plan,
        result
    )

    # ========================================================
    # STEP 4 — RETURN
    # ========================================================

    return {
        "plan": plan,
        "result": result,
        "explanation": explanation,
    }


# TODO Rename this here and in `run_analysis`
def _extracted_from_run_analysis_182(plan, df):
    group_column = plan.get(
        "group_column"
    )

    value_column = plan.get(
        "value_column"
    )

    n = plan.get(
        "n",
        10
    )

    if not group_column:
        raise ValueError(
            "No group column specified."
        )

    if not value_column:
        raise ValueError(
            "No value column specified."
        )

    return top_n(df, group_column, value_column, n)