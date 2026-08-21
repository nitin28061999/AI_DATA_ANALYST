import os
import json

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


client = genai.Client(
    api_key=API_KEY
)


MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# CHOOSE ANALYSIS
# ============================================================

def choose_analysis(
    question,
    profile
):
    """
    Ask Gemini to choose the correct Python
    analysis operation.
    """

    prompt = f"""
You are an AI Data Analyst.

Your job is to analyze a dataset by selecting
ONE operation that Python should execute.

USER QUESTION:
{question}

DATASET PROFILE:
{json.dumps(profile, indent=2, default=str)}

AVAILABLE OPERATIONS:

1. calculate_sum
Use for:
- total
- sum
- total revenue
- total sales
- total cost

Required:
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

Required:
{{
    "operation": "calculate_average",
    "column": "Selling_Price"
}}


3. calculate_count
Use for:
- count
- number of rows
- number of records
- number of transactions

Required:
{{
    "operation": "calculate_count",
    "column": "Revenue"
}}


4. calculate_min
Use for:
- minimum
- lowest
- smallest
- lowest price

Required:
{{
    "operation": "calculate_min",
    "column": "Revenue"
}}


5. calculate_max
Use for:
- maximum
- highest
- largest
- highest revenue
- highest price

Required:
{{
    "operation": "calculate_max",
    "column": "Revenue"
}}


6. group_and_sum
Use for:
- top cities by revenue
- revenue by city
- revenue by brand
- revenue by category
- sales by city
- sales by brand

Required:
{{
    "operation": "group_and_sum",
    "group_column": "City",
    "value_column": "Revenue"
}}


7. top_n
Use when the user explicitly asks for:
- top 5
- top 10
- highest 5
- best 5
- top N categories/cities/brands

Required:
{{
    "operation": "top_n",
    "group_column": "City",
    "value_column": "Revenue",
    "n": 5
}}


RULES:

- Return ONLY valid JSON.
- Do not use markdown.
- Do not explain your answer.
- Use column names exactly as they appear in the dataset.
- Never invent a column.
- Choose the simplest correct operation.
- For "top N", use top_n.
- For "by city/brand/category", use group_and_sum.
- For a simple total, use calculate_sum.

JSON:
"""


    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )


    text = response.text.strip() # pyright: ignore[reportOptionalMemberAccess]


    # --------------------------------------------------------
    # Remove markdown if Gemini accidentally adds it
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        plan = json.loads(text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Gemini returned invalid JSON: {text}"
        ) from e


    return plan


# ============================================================
# EXPLAIN RESULT
# ============================================================

def explain_result(
    question,
    plan,
    result
):
    """
    Ask Gemini to explain the Python result
    in natural language.
    """

    if hasattr(
        result,
        "to_dict"
    ):

        result_for_ai = result.to_dict(
            orient="records"
        )

    else:

        result_for_ai = result


    prompt = f"""
You are an AI Data Analyst.

Answer the user's question using the
ACTUAL Python calculation below.

USER QUESTION:
{question}

ANALYSIS PLAN:
{json.dumps(plan, indent=2, default=str)}

PYTHON RESULT:
{json.dumps(result_for_ai, indent=2, default=str)}

Instructions:

- Give a clear and concise answer.
- Use the actual Python result.
- Do not invent numbers.
- Do not perform a different calculation.
- If the result is grouped data, summarize the important findings.
- If the result contains rankings, explain the ranking.
- Format large numbers with commas.
- Use 2 decimal places for monetary/decimal values when appropriate.
- Do not mention internal prompts.
- Do not mention Gemini.
- Do not mention that you are an AI model.

Answer naturally.
"""


    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )


    return response.text.strip() # pyright: ignore[reportOptionalMemberAccess]