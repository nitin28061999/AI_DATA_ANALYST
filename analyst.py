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


# ============================================================
# RUN ANALYSIS
# ============================================================

def run_analysis(
    df,
    profile,
    question
):  # sourcery skip: low-code-quality
    """
    Run the complete AI Data Analyst workflow.

    1. Gemini understands the question.
    2. Gemini chooses an operation.
    3. Python executes the operation.
    4. Gemini explains the actual result.
    """

    # ========================================================
    # STEP 1: GEMINI CHOOSES OPERATION
    # ========================================================

    plan = choose_analysis(
        question,
        profile
    )

    if not isinstance(plan, dict):
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
    # STEP 2: EXECUTE PYTHON ANALYSIS
    # ========================================================

    # --------------------------------------------------------
    # SUM
    # --------------------------------------------------------

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

        result = _extracted_from_run_analysis_211(plan, df)
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

        result = _extracted_from_run_analysis_351(plan, df)
    elif operation == "filtered_average":

        result = _extracted_from_run_analysis_391(plan, df)
    elif operation == "filtered_count":

        result = _extracted_from_run_analysis_272(plan, df)
    elif operation == "filtered_unique_count":

        result = _extracted_from_run_analysis_303(plan, df)
    else:

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # ========================================================
    # STEP 3: GEMINI EXPLAINS RESULT
    # ========================================================

    explanation = explain_result(
        question,
        plan,
        result
    )

    # ========================================================
    # STEP 4: RETURN EVERYTHING
    # ========================================================

    return {
        "plan": plan,
        "result": result,
        "explanation": explanation,
    }


# TODO Rename this here and in `run_analysis`
def _extracted_from_run_analysis_303(plan, df):
    filter_column = plan.get(
        "filter_column"
    )

    filter_value = plan.get(
        "filter_value"
    )

    value_column = plan.get(
        "value_column"
    )

    if not filter_column:
        raise ValueError(
            "No filter column specified."
        )

    if filter_value is None:
        raise ValueError(
            "No filter value specified."
        )

    if not value_column:
        raise ValueError(
            "No value column specified."
        )

    return filtered_unique_count(df, filter_column, filter_value, value_column)


# TODO Rename this here and in `run_analysis`
def _extracted_from_run_analysis_272(plan, df):
    filter_column = plan.get(
        "filter_column"
    )

    filter_value = plan.get(
        "filter_value"
    )

    count_column = plan.get(
        "count_column"
    )

    if not filter_column:
        raise ValueError(
            "No filter column specified."
        )

    if filter_value is None:
        raise ValueError(
            "No filter value specified."
        )

    return filtered_count(df, filter_column, filter_value, count_column)


# TODO Rename this here and in `run_analysis`
def _extracted_from_run_analysis_391(plan, df):
    filter_column = plan.get(
        "filter_column"
    )

    filter_value = plan.get(
        "filter_value"
    )

    value_column = plan.get(
        "value_column"
    )

    if not filter_column:
        raise ValueError(
            "No filter column specified."
        )

    if filter_value is None:
        raise ValueError(
            "No filter value specified."
        )

    if not value_column:
        raise ValueError(
            "No value column specified."
        )

    return filtered_average(df, filter_column, filter_value, value_column)


# TODO Rename this here and in `run_analysis`
def _extracted_from_run_analysis_351(plan, df):
    filter_column = plan.get(
        "filter_column"
    )

    filter_value = plan.get(
        "filter_value"
    )

    value_column = plan.get(
        "value_column"
    )

    if not filter_column:
        raise ValueError(
            "No filter column specified."
        )

    if filter_value is None:
        raise ValueError(
            "No filter value specified."
        )

    if not value_column:
        raise ValueError(
            "No value column specified."
        )

    return filtered_sum(df, filter_column, filter_value, value_column)


# TODO Rename this here and in `run_analysis`
def _extracted_from_run_analysis_211(plan, df):
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