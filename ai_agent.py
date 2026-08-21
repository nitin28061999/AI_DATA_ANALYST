import os
import json

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


MODEL = "gemini-3.1-flash-lite"


# ============================================================
# GEMINI PLANNER
# ============================================================

def choose_analysis(
    question,
    profile
):
    """
    Ask Gemini to determine which analysis
    operation should be performed.
    """

    prompt = f"""
You are the planning component of an AI Data Analyst.

The user has uploaded a dataset.

DATASET PROFILE:

{json.dumps(
    profile,
    indent=2,
    default=str
)}

USER QUESTION:

{question}

AVAILABLE OPERATIONS:

1. calculate_sum
2. calculate_average
3. calculate_count
4. calculate_min
5. calculate_max
6. group_and_sum

Choose exactly ONE operation.

Return ONLY valid JSON.

For calculate_sum:

{{
    "operation": "calculate_sum",
    "column": "column_name"
}}

For calculate_average:

{{
    "operation": "calculate_average",
    "column": "column_name"
}}

For calculate_count:

{{
    "operation": "calculate_count",
    "column": "column_name"
}}

For calculate_min:

{{
    "operation": "calculate_min",
    "column": "column_name"
}}

For calculate_max:

{{
    "operation": "calculate_max",
    "column": "column_name"
}}

For group_and_sum:

{{
    "operation": "group_and_sum",
    "group_column": "column_name",
    "value_column": "column_name"
}}

RULES:

- Use ONLY columns that exist in the dataset.
- Never invent column names.
- Return JSON only.
- Do not use markdown.
- Do not explain your answer.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip() # pyright: ignore[reportOptionalMemberAccess]

    # Remove accidental markdown fences
    if text.startswith("```"):
        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

    try:

        plan = json.loads(text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Gemini returned invalid JSON:\n{text}"
        ) from e

    return plan


# ============================================================
# GEMINI EXPLANATION
# ============================================================

def explain_result(
    question,
    operation,
    result
):
    """
    Ask Gemini to explain the actual Python result.
    """

    if hasattr(result, "to_dict"):

        result_for_ai = result.to_dict(
            orient="records"
        )

    else:

        result_for_ai = result


    prompt = f"""
You are an AI Data Analyst.

USER QUESTION:

{question}

ANALYSIS PERFORMED:

{json.dumps(
    operation,
    indent=2,
    default=str
)}

ACTUAL RESULT FROM PYTHON:

{json.dumps(
    result_for_ai,
    indent=2,
    default=str
)}

Explain the result clearly.

RULES:

- Use ONLY the actual result provided.
- Do not invent numbers.
- Do not change numbers.
- Do not perform a different calculation.
- Mention the relevant column or columns.
- Use simple business language.
- If the result is a table, identify the most important finding.
- Keep the response concise.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text.strip() # pyright: ignore[reportOptionalMemberAccess]