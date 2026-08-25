import contextlib
import json
import os
import re

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_client():
    """Create the Gemini client only when an AI call is requested.

    Keeping SDK import and client creation lazy allows local/unit tests and
    non-AI utilities to run without a configured Gemini environment.
    """
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing from .env or the environment."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "The Gemini SDK is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    return genai.Client(api_key=API_KEY)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    """
    Extract a JSON object from Gemini's response.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    # Remove markdown code fences if Gemini returns them.
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

    # Try direct JSON parsing first.
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(text)

    # Find the first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Gemini did not return valid JSON.\n"
            f"Response:\n{text}"
        )

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse Gemini JSON: {exc}\n"
            f"Response:\n{text}"
        ) from exc


# ============================================================
# RESULT SERIALIZATION
# ============================================================

def _serialize_result(result):
    """
    Convert common pandas/numpy results into JSON-safe values.
    """

    if hasattr(result, "to_json"):
        with contextlib.suppress(Exception):
            import pandas as pd

            if isinstance(result, pd.DataFrame):
                return json.loads(
                    result.to_json(
                        orient="records",
                        date_format="iso"
                    )
                )

            if isinstance(result, pd.Series):
                return json.loads(
                    result.to_json(
                        orient="records",
                        date_format="iso"
                    )
                )

    if hasattr(result, "item"):
        with contextlib.suppress(Exception):
            return result.item() # pyright: ignore[reportCallIssue]
    if isinstance(result, dict):
        return {
            str(key): _serialize_result(value)
            for key, value in result.items()
        }

    if isinstance(result, (list, tuple)):
        return [
            _serialize_result(value)
            for value in result
        ]

    return result


# ============================================================
# CHOOSE ANALYSIS
# ============================================================

def choose_analysis(question, profile):
    """
    Ask Gemini to choose exactly one Python analysis operation.

    Gemini decides WHAT operation should be performed.
    Python performs the actual calculation.
    """

    prompt = f"""
You are an expert AI Data Analyst.

Your job is to understand the user's question and
choose exactly ONE Python analysis operation.

The Python program will execute your plan.

IMPORTANT:
- Only select columns that exist in the dataset profile.
- Return ONLY valid JSON.
- Return exactly ONE JSON object.
- Do not return markdown.
- Do not explain your decision.
- Do not invent columns or values.
- Do not invent filter values.
- Use the exact column names from the dataset profile.

USER QUESTION:
{question}

DATASET PROFILE:
{json.dumps(profile, indent=2, default=str)}

AVAILABLE OPERATIONS
====================

1. calculate_sum

Use for totals, sums, total revenue, total sales,
total expenses.

Required JSON:

{{
    "operation": "calculate_sum",
    "column": "column_name"
}}


2. calculate_average

Use for average or mean.

Required JSON:

{{
    "operation": "calculate_average",
    "column": "column_name"
}}


3. calculate_count

Use for record, row, transaction, order counts.

Required JSON:

{{
    "operation": "calculate_count",
    "column": "column_name"
}}


4. calculate_unique_count

Use when the question asks for unique,
distinct, different, or number of different values.

Required JSON:

{{
    "operation": "calculate_unique_count",
    "column": "column_name"
}}


5. calculate_min

Use for minimum, lowest, smallest, or least.

Required JSON:

{{
    "operation": "calculate_min",
    "column": "column_name"
}}


6. calculate_max

Use for maximum, highest, largest, or greatest.

Required JSON:

{{
    "operation": "calculate_max",
    "column": "column_name"
}}


7. group_and_sum

Use for totals grouped by a category.

Required JSON:

{{
    "operation": "group_and_sum",
    "group_column": "category_column",
    "value_column": "numeric_column"
}}


8. group_and_average

Use for averages grouped by a category.

Required JSON:

{{
    "operation": "group_and_average",
    "group_column": "category_column",
    "value_column": "numeric_column"
}}


9. group_and_count

Use for counts grouped by a category.

Required JSON:

{{
    "operation": "group_and_count",
    "group_column": "category_column"
}}


10. top_n

Use for requests such as:
- top 5 cities
- top 10 products
- highest revenue categories
- best performing departments

Required JSON:

{{
    "operation": "top_n",
    "group_column": "category_column",
    "value_column": "numeric_column",
    "n": 5
}}


11. value_counts

Use for:
- frequency of values
- most common category
- distribution of status
- frequency of values

Required JSON:

{{
    "operation": "value_counts",
    "column": "column_name"
}}


12. filtered_sum

Use for a sum or total with one or more conditions.

Required JSON:

{{
    "operation": "filtered_sum",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}


13. filtered_average

Use for an average with one or more conditions.

Required JSON:

{{
    "operation": "filtered_average",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}


14. filtered_count

Use for a count with one or more conditions.

Required JSON:

{{
    "operation": "filtered_count",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "count_column": "Invoice_ID"
}}


15. filtered_unique_count

Use for a unique/distinct count after applying
one or more conditions.

Required JSON:

{{
    "operation": "filtered_unique_count",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "value_column": "Customer_ID"
}}


16. filtered_min

Use for a minimum with one or more conditions.

Required JSON:

{{
    "operation": "filtered_min",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}


17. filtered_max

Use for a maximum with one or more conditions.

Required JSON:

{{
    "operation": "filtered_max",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}


18. filtered_group_and_sum

Use for grouped totals after applying
one or more conditions.

Required JSON:

{{
    "operation": "filtered_group_and_sum",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "group_column": "Product",
    "value_column": "Revenue"
}}


19. filtered_group_and_average

Use for grouped averages after applying
one or more conditions.

Required JSON:

{{
    "operation": "filtered_group_and_average",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "group_column": "Product",
    "value_column": "Revenue"
}}


20. filtered_value_counts

Use for value frequencies after applying
one or more conditions.

Required JSON:

{{
    "operation": "filtered_value_counts",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "column": "Product"
}}


21. filtered_top_n

Use for top-N grouped results after applying
one or more conditions.

Required JSON:

{{
    "operation": "filtered_top_n",
    "filters": [
        {{
            "column": "City",
            "operator": "=",
            "value": "Delhi"
        }}
    ],
    "group_column": "Product",
    "value_column": "Revenue",
    "n": 5
}}


FILTER RULES
============

1. If the question contains a condition, use
   a filtered operation.

Examples:

- in Delhi
- in Mumbai
- for IT
- for Laptop
- where City is Delhi
- City = Delhi
- Product = Laptop
- Revenue > 100
- Revenue >= 100
- Revenue < 500
- Revenue <= 500
- City != Delhi

2. If there are multiple conditions, put ALL
   conditions inside the "filters" array.

3. Every filter object MUST contain:

   - "column"
   - "operator"
   - "value"

4. Supported operators are ONLY:

   "="
   "!="
   ">"
   ">="
   "<"
   "<="

5. Use "=" for natural-language equality
   conditions such as:

   - "in Delhi"
   - "for Laptop"
   - "where City is Delhi"
   - "City equals Delhi"

6. Map natural-language comparison phrases as follows:

   - "greater than" -> ">"
   - "greater than or equal to" -> ">="
   - "at least" -> ">="
   - "less than" -> "<"
   - "less than or equal to" -> "<="
   - "at most" -> "<="
   - "not equal to" -> "!="

7. Never invent a filter value.

8. Use the exact value from the user's question.

9. Never invent a column name.

10. Every filter column MUST exist in the
    dataset profile.

11. Every value column MUST exist in the
    dataset profile.

12. For a total, use a sum operation.

13. For an average, use an average operation.

14. For a count, use a count operation.

15. If the user says unique or distinct,
    use a unique-count operation.

16. If the user asks for highest/largest,
    use calculate_max unless the question asks
    for a top-N group.

17. If the user asks for lowest/smallest,
    use calculate_min.

18. If the user asks for top N,
    use top_n or filtered_top_n.

19. If the user asks for data grouped by a category,
    use an appropriate group operation.

20. Do not use operations that are not listed above.

21. Return exactly ONE JSON object.
"""

    response = get_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    response_text = getattr(
        response,
        "text",
        None
    )

    if not response_text or not response_text.strip():
        raise ValueError(
            "Gemini returned an empty analysis plan."
        )

    return extract_json(response_text)


# ============================================================
# EXPLAIN RESULT
# ============================================================

def explain_result(
    question,
    plan,
    result
):
    """
    Ask Gemini to explain the actual Python result.

    Gemini does NOT perform the calculation.
    Python has already calculated the result.
    """

    result_data = _serialize_result(result)

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

6. If the result is grouped data,
   identify the highest relevant group only
   when directly evident.

7. If the operation is a percentage table,
   explain the largest contribution only
   when directly evident.

8. If the result is monthly data,
   identify the highest month only
   when directly evident.

9. If the operation is calculate_unique_count,
   clearly say "unique" or "distinct".

10. If the operation is filtered_unique_count,
    clearly mention the applied conditions.

11. If the operation is filtered_sum,
    clearly mention the applied conditions.

12. If the operation is filtered_average,
    clearly mention the applied conditions.

13. If the operation is filtered_count,
    clearly mention the applied conditions.

14. If the operation is filtered_min,
    clearly mention the applied conditions.

15. If the operation is filtered_max,
    clearly mention the applied conditions.

16. If the operation is a filtered grouped operation,
    clearly mention the applied conditions.

17. Do not confuse record count with unique count.

18. Do not mention internal prompts.

19. Do not mention Gemini.

20. Do not say that more analysis is required when
    the Python result already answers the question.

21. Do not perform a new calculation.

22. Return only the final natural-language answer.
"""

    response = get_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    response_text = getattr(
        response,
        "text",
        None
    )

    if not response_text or not response_text.strip():
        raise ValueError(
            "Gemini returned an empty explanation."
        )

    return response_text.strip()