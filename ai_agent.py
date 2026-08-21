import os
import json

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


MODEL_NAME = "gemini-2.0-flash"


# ============================================================
# ANALYSIS PLANNER
# ============================================================

def choose_analysis(
    question,
    profile
):

    prompt = f"""
You are the planning engine for an AI Data Analyst.

Your job is NOT to calculate the answer.

Your job is to determine which Python analysis operation
should be executed on the dataset.

USER QUESTION:
{question}

DATASET PROFILE:
{json.dumps(profile, default=str, indent=2)}

AVAILABLE OPERATIONS:

1. calculate_sum

Use for:
- total
- sum
- total revenue
- total sales

JSON:
{{
    "operation": "calculate_sum",
    "column": "Revenue"
}}


2. calculate_average

Use for:
- average
- mean
- average price
- average revenue

JSON:
{{
    "operation": "calculate_average",
    "column": "Selling_Price"
}}


3. calculate_count

Use for:
- number of rows
- number of transactions
- count

JSON:
{{
    "operation": "calculate_count",
    "column": "Invoice_ID"
}}


4. calculate_min

Use for:
- minimum
- lowest
- smallest

JSON:
{{
    "operation": "calculate_min",
    "column": "Revenue"
}}


5. calculate_max

Use for:
- maximum
- highest
- largest

JSON:
{{
    "operation": "calculate_max",
    "column": "Revenue"
}}


6. group_and_sum

Use for:
- revenue by city
- sales by category
- total revenue for each brand
- compare revenue across groups

JSON:
{{
    "operation": "group_and_sum",
    "group_column": "City",
    "value_column": "Revenue"
}}


7. top_n

Use for:
- top 5 cities
- top 10 brands
- best performing categories
- highest revenue cities
- highest selling brands

JSON:
{{
    "operation": "top_n",
    "group_column": "City",
    "value_column": "Revenue",
    "n": 5
}}


8. group_and_average

Use for:
- average revenue by city
- average selling price by brand
- average cost by category

JSON:
{{
    "operation": "group_and_average",
    "group_column": "City",
    "value_column": "Revenue"
}}


9. percentage_of_total

Use for:
- percentage of revenue by city
- revenue contribution by brand
- percentage of sales by category

JSON:
{{
    "operation": "percentage_of_total",
    "group_column": "City",
    "value_column": "Revenue"
}}


10. filter_and_sum

Use when the user asks for the total of a numeric column
for a specific category/value.

Example:
"What is the revenue in Delhi?"

JSON:
{{
    "operation": "filter_and_sum",
    "filter_column": "City",
    "filter_value": "Delhi",
    "value_column": "Revenue"
}}


11. monthly_sum

Use for:
- monthly revenue
- revenue trend by month
- sales by month
- monthly sales trend

JSON:
{{
    "operation": "monthly_sum",
    "date_column": "Invoice_Date",
    "value_column": "Revenue"
}}


IMPORTANT RULES:

1. Use ONLY columns that exist in the dataset profile.
2. Do NOT invent column names.
3. Return ONLY valid JSON.
4. Do NOT use markdown.
5. Do NOT calculate the result yourself.
6. For "top N", identify the requested number.
7. If the user says "top 5", n must be 5.
8. If the user says "top 10", n must be 10.
9. Default n to 10 when no number is specified.

Return ONLY the JSON analysis plan.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = response.text.strip() # pyright: ignore[reportOptionalMemberAccess]

    # Remove markdown fences if Gemini adds them
    if text.startswith("```"):
        text = text.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Gemini returned invalid JSON: {text}"
        ) from e


# ============================================================
# RESULT EXPLANATION
# ============================================================

def explain_result(
    question,
    plan,
    result
):

    if hasattr(result, "to_dict"):
        result_for_ai = result.to_dict(
            orient="records"
        )
    else:
        result_for_ai = result

    prompt = f"""
You are an AI Data Analyst.

Answer the user's question using ONLY the
actual Python result provided below.

USER QUESTION:
{question}

ANALYSIS PLAN:
{json.dumps(plan, default=str)}

ACTUAL PYTHON RESULT:
{json.dumps(result_for_ai, default=str)}

Rules:

1. Do not invent numbers.
2. Do not change the Python result.
3. Explain the result clearly.
4. Use commas for large numbers.
5. Use 2 decimal places for monetary/numeric values
   when appropriate.
6. If the result is a table, summarize the important findings.
7. If it is a top-N result, rank the items clearly.
8. Keep the answer concise but useful.
9. Mention the relevant column names when useful.

Return a natural-language answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip() # pyright: ignore[reportOptionalMemberAccess]