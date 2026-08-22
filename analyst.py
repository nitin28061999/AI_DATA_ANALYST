# analyst.py

from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai.operations import Operations
from google.genai.types import Operation

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
    dataframe_to_records,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(
    text: str
) -> dict:
    """
    Extract a JSON object from Gemini's response.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    with contextlib.suppress(
        json.JSONDecodeError
    ):

        result = json.loads(
            text
        )

        if isinstance(result, dict):
            return result

    # --------------------------------------------------------
    # JSON embedded inside text
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):

        raise ValueError(
            "Gemini did not return valid JSON.\n"
            f"Response:\n{text}"
        )

    json_text = text[
        start:end + 1
    ]

    try:

        result = json.loads(
            json_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Could not parse Gemini JSON: {exc}\n"
            f"Response:\n{text}"
        ) from exc

    if not isinstance(result, dict):

        raise ValueError(
            "Gemini JSON response must be an object."
        )

    return result


# ============================================================
# PROFILE HELPERS
# ============================================================

def _profile_columns(
    profile: dict
) -> list[str]:
    """
    Extract column names from different profile formats.
    """

    if not isinstance(
        profile,
        dict
    ):
        raise ValueError(
            "Dataset profile must be a dictionary."
        )

    columns = profile.get(
        "columns"
    )

    if isinstance(
        columns,
        list
    ):
        return [
            str(column)
            for column in columns
        ]

    if isinstance(
        columns,
        dict
    ):
        return [
            str(column)
            for column in columns.keys()
        ]

    # Some profiles may use column_info.
    column_info = profile.get(
        "column_info"
    )

    if isinstance(
        column_info,
        dict
    ):
        return [
            str(column)
            for column in column_info.keys()
        ]

    if isinstance(
        column_info,
        list
    ):
        return [
            str(item["name"])
            for item in column_info
            if isinstance(item, dict) and "name" in item
        ]
    return []


def _column_exists(
    profile: dict,
    column: Any
) -> bool:

    return (
        column in _profile_columns(profile)
    )


# ============================================================
# NORMALIZE FILTERS
# ============================================================

def normalize_filters(
    filters,
    profile: dict
) -> list:
    """
    Normalize Gemini-generated filters.

    Gemini may return:

        {
            "column": "City",
            "value": "Delhi"
        }

    This becomes:

        {
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }

    The function also validates columns and operators.
    """

    if filters is None:
        return []

    if not isinstance(
        filters,
        list
    ):
        raise ValueError(
            "filters must be a list."
        )

    columns = _profile_columns(
        profile
    )

    supported_operators = {
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

    normalized = []

    for index, item in enumerate(filters):

        if not isinstance(
            item,
            dict
        ):
            raise ValueError(
                f"Filter #{index + 1} must be an object."
            )

        if "column" not in item:
            raise ValueError(
                f"Filter #{index + 1} is missing 'column'."
            )

        if "value" not in item:
            raise ValueError(
                f"Filter #{index + 1} is missing 'value'."
            )

        column = item["column"]

        if columns and column not in columns:

            raise ValueError(
                f"Filter column '{column}' does not exist "
                f"in the dataset."
            )

        operator = item.get(
            "operator",
            "="
        )

        if operator is None:
            operator = "="

        operator = str(
            operator
        ).strip().lower()

        if operator not in supported_operators:

            raise ValueError(
                f"Unsupported filter operator "
                f"'{operator}'."
            )

        if operator == "==":
            operator = "="

        value = item["value"]

        if operator == "between":

            if not isinstance(
                value,
                (list, tuple)
            ):

                raise ValueError(
                    "The 'between' filter requires "
                    "two values."
                )

            if len(value) != 2:

                raise ValueError(
                    "The 'between' filter requires "
                    "exactly two values."
                )

            value = [
                value[0],
                value[1]
            ]

        normalized.append(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

    return normalized


# ============================================================
# VALIDATE PLAN
# ============================================================

def validate_plan(
    plan: dict,
    profile: dict
) -> dict:  # sourcery skip: low-code-quality
    """
    Validate and normalize Gemini's analysis plan.
    """

    if not isinstance(
        plan,
        dict
    ):
        raise ValueError(
            "Analysis plan must be a JSON object."
        )

    operation = plan.get(
        "operation"
    )

    if not operation:
        raise ValueError(
            "Analysis plan is missing 'operation'."
        )

    operation = str(
        operation
    ).strip()

    allowed_operations = {
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

    if operation not in allowed_operations:

        raise ValueError(
            f"Unsupported analysis operation "
            f"'{operation}'."
        )

    plan = dict(
        plan
    )

    plan["operation"] = operation

    columns = _profile_columns(
        profile
    )

    # ========================================================
    # COLUMN OPERATIONS
    # ========================================================

    column_operations = {
        "calculate_sum",
        "calculate_average",
        "calculate_count",
        "calculate_unique_count",
        "calculate_min",
        "calculate_max",
        "value_counts",
    }

    if operation in column_operations:

        column = plan.get(
            "column"
        )

        if not column:

            raise ValueError(
                f"{operation} requires 'column'."
            )

        if columns and column not in columns:

            raise ValueError(
                f"Column '{column}' does not exist "
                f"in the dataset."
            )

    # ========================================================
    # GROUP + VALUE OPERATIONS
    # ========================================================

    group_operations = {
        "group_and_sum",
        "group_and_average",
        "top_n",
        "percentage_of_total",
        "filtered_group_and_sum",
        "filtered_group_and_average",
        "filtered_top_n",
    }

    if operation in group_operations:

        group_column = plan.get(
            "group_column"
        )

        value_column = plan.get(
            "value_column"
        )

        if not group_column:

            raise ValueError(
                f"{operation} requires 'group_column'."
            )

        if not value_column:

            raise ValueError(
                f"{operation} requires 'value_column'."
            )

        if columns and group_column not in columns:

            raise ValueError(
                f"Group column '{group_column}' "
                f"does not exist."
            )

        if columns and value_column not in columns:

            raise ValueError(
                f"Value column '{value_column}' "
                f"does not exist."
            )

    # ========================================================
    # GROUP COUNT
    # ========================================================

    if operation == "group_and_count":
        group_column = plan.get(
            "group_column"
        )

        if not group_column:

            raise ValueError(
                "group_and_count requires "
                "'group_column'."
            )

        if columns and group_column not in columns:

            raise ValueError(
                f"Group column '{group_column}' "
                f"does not exist."
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
                "monthly_sum requires "
                "'date_column'."
            )

        if not value_column:

            raise ValueError(
                "monthly_sum requires "
                "'value_column'."
            )

        if columns and date_column not in columns:

            raise ValueError(
                f"Date column '{date_column}' "
                f"does not exist."
            )

        if columns and value_column not in columns:

            raise ValueError(
                f"Value column '{value_column}' "
                f"does not exist."
            )

    # ========================================================
    # FILTERED OPERATIONS
    # ========================================================

    filtered_operations = {
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

    if operation in filtered_operations:

        filters = plan.get(
            "filters",
            []
        )

        if not filters:

            raise ValueError(
                f"{operation} requires at least "
                "one filter."
            )

        plan["filters"] = normalize_filters(
            filters,
            profile
        )

    # ========================================================
    # FILTERED VALUE COLUMN
    # ========================================================

    filtered_value_operations = {
        "filtered_sum",
        "filtered_average",
        "filtered_unique_count",
        "filtered_min",
        "filtered_max",
        "filtered_group_and_sum",
        "filtered_group_and_average",
        "filtered_top_n",
    }

    if operation in filtered_value_operations:

        value_column = plan.get(
            "value_column"
        )

        if not value_column:

            raise ValueError(
                f"{operation} requires "
                "'value_column'."
            )

        if columns and value_column not in columns:

            raise ValueError(
                f"Value column '{value_column}' "
                f"does not exist."
            )

    # ========================================================
    # FILTERED COUNT
    # ========================================================

    if operation == "filtered_count":

        count_column = plan.get(
            "count_column"
        )

        # If Gemini doesn't provide count_column,
        # fall back to the first filter column.
        if not count_column:

            if plan["filters"]:

                count_column = (
                    plan["filters"][0]["column"]
                )

            else:

                raise ValueError(
                    "filtered_count requires "
                    "'count_column'."
                )

            plan["count_column"] = count_column

        if columns and count_column not in columns:

            raise ValueError(
                f"Count column '{count_column}' "
                f"does not exist."
            )

    # ========================================================
    # FILTERED GROUP
    # ========================================================

    filtered_group_operations = {
        "filtered_group_and_sum",
        "filtered_group_and_average",
        "filtered_top_n",
    }

    if operation in filtered_group_operations:

        group_column = plan.get(
            "group_column"
        )

        if not group_column:

            raise ValueError(
                f"{operation} requires "
                "'group_column'."
            )

        if columns and group_column not in columns:

            raise ValueError(
                f"Group column '{group_column}' "
                f"does not exist."
            )

    # ========================================================
    # FILTERED VALUE COUNTS
    # ========================================================

    if operation == "filtered_value_counts":

        column = plan.get(
            "column"
        )

        if not column:

            raise ValueError(
                "filtered_value_counts requires "
                "'column'."
            )

        if columns and column not in columns:

            raise ValueError(
                f"Column '{column}' does not exist."
            )

    # ========================================================
    # TOP N
    # ========================================================

    if operation in {
        "top_n",
        "filtered_top_n",
    }:

        n = plan.get(
            "n",
            5
        )

        try:
            n = int(n)
        except (
            ValueError,
            TypeError
        ) as exc:

            raise ValueError(
                "n must be an integer."
            ) from exc

        if n <= 0:

            raise ValueError(
                "n must be greater than zero."
            )

        plan["n"] = n

    return plan


# ============================================================
# CHOOSE ANALYSIS
# ============================================================

def choose_analysis(
    question: str,
    profile: dict
) -> dict:
    """
    Ask Gemini to choose exactly one analysis operation.

    Gemini decides WHAT should be calculated.

    Python performs the actual calculation.
    """

    prompt = f"""
You are an expert AI Data Analyst.

Your job is to understand the user's question and
choose exactly ONE Python analysis operation.

The Python program will execute your plan.

IMPORTANT:

1. Only select columns that exist in the dataset profile.
2. Never invent column names.
3. Never invent filter values.
4. If a condition is present, include it in filters.
5. Filters may omit operator; if omitted, Python treats it as "=".
6. Return ONLY valid JSON.
7. Return exactly ONE JSON object.
8. Do not use markdown.
9. Do not explain your decision.

USER QUESTION:
{question}

DATASET PROFILE:
{json.dumps(profile, indent=2, default=str)}


AVAILABLE OPERATIONS
====================

1. calculate_sum

Use for:
- total
- sum
- total revenue
- total sales
- total expenses

JSON:
{{
    "operation": "calculate_sum",
    "column": "column_name"
}}


2. calculate_average

Use for:
- average
- mean
- average salary
- average price
- average score

JSON:
{{
    "operation": "calculate_average",
    "column": "column_name"
}}


3. calculate_count

Use for:
- count records
- number of rows
- number of transactions
- number of invoices

JSON:
{{
    "operation": "calculate_count",
    "column": "column_name"
}}


4. calculate_unique_count

Use for:
- unique customers
- distinct customers
- unique products
- distinct employees
- different cities
- unique IDs

JSON:
{{
    "operation": "calculate_unique_count",
    "column": "column_name"
}}


5. calculate_min

Use for:
- minimum
- lowest
- smallest
- least

JSON:
{{
    "operation": "calculate_min",
    "column": "column_name"
}}


6. calculate_max

Use for:
- maximum
- highest
- largest
- greatest

JSON:
{{
    "operation": "calculate_max",
    "column": "column_name"
}}


7. group_and_sum

Use for:
- revenue by city
- sales by product
- expenses by department
- salary by department

JSON:
{{
    "operation": "group_and_sum",
    "group_column": "category_column",
    "value_column": "numeric_column"
}}


8. group_and_average

Use for:
- average salary by department
- average price by category
- average score by class

JSON:
{{
    "operation": "group_and_average",
    "group_column": "category_column",
    "value_column": "numeric_column"
}}


9. group_and_count

Use for:
- number of customers by city
- orders by product
- employees by department
- transactions by category

JSON:
{{
    "operation": "group_and_count",
    "group_column": "category_column"
}}


10. top_n

Use for:
- top 5 cities
- top 10 products
- highest revenue categories
- best performing departments

JSON:
{{
    "operation": "top_n",
    "group_column": "category_column",
    "value_column": "numeric_column",
    "n": 5
}}


11. percentage_of_total

Use for:
- percentage of revenue by city
- contribution by category
- share of sales
- percentage of total

JSON:
{{
    "operation": "percentage_of_total",
    "group_column": "category_column",
    "value_column": "numeric_column"
}}


12. monthly_sum

Use for:
- monthly revenue
- sales by month
- monthly expenses
- revenue trend over time

JSON:
{{
    "operation": "monthly_sum",
    "date_column": "date_column",
    "value_column": "numeric_column"
}}


13. value_counts

Use for:
- most common category
- frequency of values
- status distribution
- frequency of values

JSON:
{{
    "operation": "value_counts",
    "column": "column_name"
}}


14. filtered_sum

Use when asking for a total/sum with conditions.

Example:
"total revenue in Delhi for Laptop"

JSON:
{{
    "operation": "filtered_sum",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }},
        {{
            "column": "Product",
    if operation in {
            "calculate_sum",
            "calculate_average",
            "calculate_count",
            "calculate_unique_count",
            "calculate_min",
            "calculate_max",
            "value_counts",
        } and not plan.get("column"):
        raise ValueError(
            f"{Operation} requires 'column'."
        )== "group_and_count" and not plan.get("group_column"):
        raise ValueError(
            "group_and_count requires "
            "'group_column'."
        )
    "operation": "filtered_count",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "count_column": "Invoice_ID"
}}


17. filtered_unique_count

Use if operation == "group_and_count" and not plan.get("group_column"):
        raise ValueError(
            "group_and_count requires "
            "'group_column'."
        )
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "value_column": "Customer_ID"
}}


18. filtered_min

Use when asking for minimum after conditions.

JSON:
{{
    "operation": "filtered_min",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}


19. filtered_max

Use when asking for maximum after conditions.

JSON:
{{
    "operation": "filtered_max",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}


20. filtered_group_and_sum

Use when conditions are applied before grouping and summing.

JSON:
{{ifif operation in {
            "filtered_group_and_sum",
            "filtered_group_and_average",
            "filtered_top_n",
        } and not plan.get("filters"):
        raise ValueError(
            f"{Operation} requires "
            "'filters'."
        )
    ],
    "group_column": "Product",
    "value_column": "Revenue"
}}


22. filtered_value_counts

Use when conditions are applied before counting value frequencies.
if operation in {
            "filtered_group_and_sum",
            "filtered_group_and_average",
            "filtered_top_n",
        } and not plan.get("filters"):
        raise ValueError(
            f"{Operation} requires "
            "'filters'."
        ) "Product"
}}
if operation in {
            "filtered_group_and_sum",
            "filtered_group_and_average",
            "filtered_top_n",
        } and not plan.get("filters"):
        raise ValueError(
            f"{Operation} requires "
            "'filters'."
        )
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "group_column": "Product",
    "value_column": "Revenue",
    "n": 5
}}


FILTER RULES
============

If the user says:

- in Delhi
- in Mumbai
- for IT
- for Laptop
- where City is Delhi
- City = Delhi
- Revenue > 1000
- Price <= 500
- Product contains Laptop
- salary between 50000 and 70000

then create a filter.

Supported operators:

=
==
!=
>
>=
<
<=
contains
between

For simple equality filters you may omit operator.

For example:

{{
    "column": "City",
    "value": "Delhi"
}}

means:

{{
    "column": "City",
    "operator": "=",
    "value": "Delhi"
}}

For comparisons, explicitly include the operator.

Example:

{{
    "column": "Revenue",
    "operator": ">",
    "value": 1000
}}

For between:

{{
    "column": "Salary",
    "operator": "between",
    "value": [50000, 70000]
}}


IMPORTANT:

If there are multiple conditions, include ALL of them.

Never invent columns.

Never invent filter values.

Return exactly ONE JSON object.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = getattr(
        response,
        "text",
        None
    )

    plan = extract_json(
        text # pyright: ignore[reportArgumentType]
    )

    return validate_plan(
        plan,
        profile
    )


# ============================================================
# EXECUTE PLAN
# ============================================================

def execute_plan(
    df,
    plan: dict
):
    """
    Execute the validated analysis plan.

    Gemini does not calculate anything.
    Python performs the actual calculation.
    """

    operation = plan[
        "operation"
    ]

    # ========================================================
    # BASIC
    # ========================================================

    if operation == "calculate_sum":

        return calculate_sum(
            df,
            plan["column"]
        )

    if operation == "calculate_average":

        return calculate_average(
            df,
            plan["column"]
        )

    if operation == "calculate_count":

        return calculate_count(
            df,
            plan["column"]
        )

    if operation == "calculate_unique_count":

        return calculate_unique_count(
            df,
            plan["column"]
        )

    if operation == "calculate_min":

        return calculate_min(
            df,
            plan["column"]
        )

    if operation == "calculate_max":

        return calculate_max(
            df,
            plan["column"]
        )

    # ========================================================
    # GROUP
    # ========================================================

    if operation == "group_and_sum":

        return group_and_sum(
            df,
            plan["group_column"],
            plan["value_column"]
        )

    if operation == "group_and_average":

        return group_and_average(
            df,
            plan["group_column"],
            plan["value_column"]
        )

    if operation == "group_and_count":

        return group_and_count(
            df,
            plan["group_column"]
        )

    # ========================================================
    # TOP N
    # ========================================================

    if operation == "top_n":

        return top_n(
            df,
            plan["group_column"],
            plan["value_column"],
            plan.get("n", 5)
        )

    # ========================================================
    # PERCENTAGE
    # ========================================================

    if operation == "percentage_of_total":

        return percentage_of_total(
            df,
            plan["group_column"],
            plan["value_column"]
        )

    # ========================================================
    # MONTHLY
    # ========================================================

    if operation == "monthly_sum":

        return monthly_sum(
            df,
            plan["date_column"],
            plan["value_column"]
        )

    # ========================================================
    # VALUE COUNTS
    # ========================================================

    if operation == "value_counts":

        return value_counts(
            df,
            plan["column"]
        )

    # ========================================================
    # FILTERED SUM
    # ========================================================

    if operation == "filtered_sum":

        return filtered_sum(
            df,
            plan["filters"],
            plan["value_column"]
        )

    # ========================================================
    # FILTERED AVERAGE
    # ========================================================

    if operation == "filtered_average":

        return filtered_average(
            df,
            plan["filters"],
            plan["value_column"]
        )

    # ========================================================
    # FILTERED COUNT
    # ========================================================

    if operation == "filtered_count":

        return filtered_count(
            df,
            plan["filters"],
            plan["count_column"]
        )

    # ========================================================
    # FILTERED UNIQUE
    # ========================================================

    if operation == "filtered_unique_count":

        return filtered_unique_count(
            df,
            plan["filters"],
            plan["value_column"]
        )

    # ========================================================
    # FILTERED MIN
    # ========================================================

    if operation == "filtered_min":

        return filtered_min(
            df,
            plan["filters"],
            plan["value_column"]
        )

    # ========================================================
    # FILTERED MAX
    # ========================================================

    if operation == "filtered_max":

        return filtered_max(
            df,
            plan["filters"],
            plan["value_column"]
        )

    # ========================================================
    # FILTERED GROUP SUM
    # ========================================================

    if operation == "filtered_group_and_sum":

        return filtered_group_and_sum(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"]
        )

    # ========================================================
    # FILTERED GROUP AVERAGE
    # ========================================================

    if operation == "filtered_group_and_average":

        return filtered_group_and_average(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"]
        )

    # ========================================================
    # FILTERED VALUE COUNTS
    # ========================================================

    if operation == "filtered_value_counts":

        return filtered_value_counts(
            df,
            plan["filters"],
            plan["column"]
        )

    # ========================================================
    # FILTERED TOP N
    # ========================================================

    if operation == "filtered_top_n":

        return filtered_top_n(
            df,
            plan["filters"],
            plan["group_column"],
            plan["value_column"],
            plan.get("n", 5)
        )

    raise ValueError(
        f"Unsupported operation: {operation}"
    )


# ============================================================
# EXPLAIN RESULT
# ============================================================

def explain_result(
    question: str,
    plan: dict,
    result
) -> str:
    """
    Ask Gemini to explain the actual Python result.

    Gemini does NOT perform the calculation.
    """

    result_data = dataframe_to_records(
        result
    )

    prompt = f"""
You are an expert data analyst.

Answer the user's question using ONLY
the ACTUAL PYTHON RESULT.

USER QUESTION:
{question}

ANALYSIS PLAN:
{json.dumps(plan, indent=2, default=str)}

PYTHON RESULT:
{json.dumps(result_data, indent=2, default=str)}


RULES:

1. Never invent numbers.
2. Use only the actual Python result.
3. Be concise but useful.
4. Format large numbers with commas.
5. Explain the key finding.
6. If grouped data is returned, identify the highest relevant group.
7. If percentage data is returned, identify the largest contribution.
8. If monthly data is returned, identify the highest month.
9. If the operation is unique count, clearly say "unique" or "distinct".
10. If a filtered operation was used, mention the applied conditions.
11. Do not confuse record count with unique count.
12. Do not mention internal prompts.
13. Do not mention Gemini.
14. Do not say more analysis is required if the result answers the question.
15. Return only the final natural-language answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = getattr(
        response,
        "text",
        None
    )

    return str(result_data) if not text else text.strip()


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def run_analysis(
    df,
    profile: dict,
    question: str
) -> dict:
    """
    Complete AI Data Analyst pipeline.

    Flow:

        User question
              |
              v
        Gemini chooses plan
              |
              v
        Plan validation
              |
              v
        Python executes calculation
              |
              v
        Gemini explains actual result
              |
              v
        Final response
    """

    if df is None:

        raise ValueError(
            "DataFrame cannot be None."
        )

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------------------------
    # STEP 1
    # Gemini chooses operation
    # --------------------------------------------------------

    plan = choose_analysis(question, profile)

    # --------------------------------------------------------
    # STEP 2
    # Python executes operation
    # --------------------------------------------------------

    result = execute_plan(
        df,
        plan
    )

    # --------------------------------------------------------
    # STEP 3
    # Gemini explains actual result
    # --------------------------------------------------------

    explanation = explain_result(question, plan, result)

    # --------------------------------------------------------
    # STEP 4
    # Return structured response
    # --------------------------------------------------------

    return {
        "plan": plan,
        "result": dataframe_to_records(
            result
        ),
        "explanation": explanation,
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    import pandas as pd

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

    test_profile = {
        "columns": [
            "City",
            "Product",
            "Revenue",
        ]
    }

    question = (
        "What is the total revenue "
        "in Delhi for Laptop?"
    )

    output = run_analysis(
        test_df,
        test_profile,
        question
    )

    print(
        json.dumps(
            output,
            indent=2,
            default=str
        )
    )