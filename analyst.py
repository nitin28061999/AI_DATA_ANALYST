from ai_agent import (
    choose_analysis,
    explain_result,
)

from analysis_tools import (
    calculate_sum,
    calculate_average,
    calculate_count,
    calculate_min,
    calculate_max,
    group_and_sum,
)


def run_analysis(
    df,
    profile,
    question
):
    """
    Run the complete AI Data Analyst workflow.
    """

    # ========================================================
    # STEP 1: GEMINI CHOOSES OPERATION
    # ========================================================

    plan = choose_analysis(
        question,
        profile
    )


    # ========================================================
    # STEP 2: VALIDATE PLAN
    # ========================================================

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
    # STEP 3: EXECUTE PYTHON ANALYSIS
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


    else:

        raise ValueError(
            f"Unsupported operation: {operation}"
        )


    # ========================================================
    # STEP 4: GEMINI EXPLAINS RESULT
    # ========================================================

    explanation = explain_result(
        question,
        plan,
        result
    )


    # ========================================================
    # STEP 5: RETURN EVERYTHING
    # ========================================================

    return {
        "plan": plan,
        "result": result,
        "explanation": explanation,
    }