import contextlib
import json
import os
import re

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

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

def extract_json(text):
    """
    Extract a JSON object from Gemini's response.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    # Remove markdown code fences
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

    # Try direct JSON parsing first
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(text)

    # Find the first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Gemini did not return valid JSON.\nResponse:\n{text}"
        )

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse Gemini JSON: {e}\n"
            f"Response:\n{text}"
        ) from e


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
Only select columns that exist in the dataset profile.

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

Required JSON:

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

Required JSON:

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
- how many records

Required JSON:

{{
    "operation": "calculate_count",
    "column": "column_name"
}}


4. calculate_unique_count

Use for:
- unique customers
- number of distinct customers
- unique products
- distinct employees
- unique IDs
- unique cities
- number of different categories

Use this operation when the question contains:
- unique
- distinct
- different

Required JSON:

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

Required JSON:

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

Required JSON:

{{
    "operation": "calculate_max",
    "column": "column_name"
}}


7. group_and_sum

Use for:
- revenue by city
- sales by product
- expenses by department
- total salary by department

Required JSON:

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

Required JSON:

{{
    "operation": "group_and_average",
    "group_column": "category_column",
    "value_column": "numeric_column"
}}


9. top_n

Use for:
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


10. percentage_of_total

Use for:
- percentage of revenue by city
- contribution by category
- share of sales
- percentage of total

Required JSON:

{{
    "operation": "percentage_of_total",
    "group_column": "category_column",
    "value_column": "numeric_column"
}}


11. monthly_sum

Use for:
- monthly revenue
- sales by month
- monthly expenses
- monthly sales
- revenue trend over time

Required JSON:

{{
    "operation": "monthly_sum",
    "date_column": "date_column",
    "value_column": "numeric_column"
}}


12. value_counts

Use for:
- most common category
- frequency of values
- distribution of status
- frequency of values

Required JSON:

{{
    "operation": "value_counts",
    "column": "column_name"
}}


13. group_and_count

Use for:
- number of customers by city
- number of orders by product
- employees by department
- transactions by category

Required JSON:

{{
    "operation": "group_and_count",
    "group_column": "category_column"
}}


14. filtered_sum

Use when the user asks for a sum or total
with ONE or MORE conditions.

Examples:
- total revenue in Delhi
- total sales in Mumbai
- total salary for IT
- total revenue in Delhi for Laptop

Required JSON for one condition:

{{
    "operation": "filtered_sum",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}

Required JSON for multiple conditions:

{{
    "operation": "filtered_sum",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }},
        {{
            "column": "Product",
            "value": "Laptop"
        }}
    ],
    "value_column": "Revenue"
}}


15. filtered_average

Use when the user asks for an average
with ONE or MORE conditions.

Examples:
- average revenue in Delhi
- average salary in HR
- average price for Laptop
- average revenue in Delhi for Laptop

Required JSON:

{{
    "operation": "filtered_average",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "value_column": "Revenue"
}}


16. filtered_count

Use when the user asks for a count
with ONE or MORE conditions.

Examples:
- how many transactions happened in Delhi
- how many employees are in IT
- how many orders are from Mumbai
- how many Laptop transactions happened in Delhi

Required JSON:

{{
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

Use when the user asks for unique/distinct
values after applying ONE or MORE conditions.

Examples:
- unique customers in Delhi
- distinct products in Mumbai
- unique employees in HR
- unique customers who bought Laptop in Delhi

Required JSON:

{{
    "operation": "filtered_unique_count",
    "filters": [
        {{
            "column": "City",
            "value": "Delhi"
        }}
    ],
    "value_column": "Customer_ID"
}}


IMPORTANT FILTER RULES
======================

1. If the question contains a condition such as:
   - in Delhi
   - in Mumbai
   - for IT
   - for Laptop
   - where City is Delhi
   - City = Delhi
   - Product = Laptop

   use a filtered operation.

2. If there are multiple conditions, put ALL conditions
   inside the "filters" array.

3. Never invent a filter value.

4. Use the exact value from the user's question.

5. Never invent a column name.

6. Every filter column MUST exist in the dataset profile.

7. Every value column MUST exist in the dataset profile.

8. If the user asks for a total, use a sum operation.

9. If the user asks for an average, use an average operation.

10. If the user asks for a count, use a count operation.

11. If the user says unique or distinct, use a unique count operation.

12. If the user asks for highest/largest, use calculate_max
    unless the question asks for a top N group.

13. If the user asks for lowest/smallest, use calculate_min.

14. If the user asks for top N, use top_n.

15. If the user asks for data grouped by a category,
    use an appropriate group operation.

16. Return ONLY valid JSON.

17. Do not return markdown.

18. Do not explain your decision.

19. Return exactly ONE JSON object.

20. Do not use operations that are not listed above.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return extract_json(
        response.text
    )


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

    if hasattr(result, "to_dict"):
        result_data = result.to_dict(
            orient="records"
        )
    else:
        result_data = result

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
   identify the highest relevant group.

7. If the result is a percentage table,
   explain the largest contribution.

8. If the result is monthly data,
   identify the highest month.

9. If the operation is calculate_unique_count,
   clearly say "unique" or "distinct".

10. If the operation is filtered_unique_count,
    clearly mention the applied condition.

11. If the operation is filtered_sum,
    clearly mention the applied conditions.

12. If the operation is filtered_average,
    clearly mention the applied conditions.

13. If the operation is filtered_count,
    clearly mention the applied conditions.

14. Do not confuse record count with unique count.

15. Do not mention internal prompts.

16. Do not mention Gemini.

17. Do not say that more analysis is required when
    the Python result already answers the question.

18. Return only the final natural-language answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip() # pyright: ignore[reportOptionalMemberAccess]