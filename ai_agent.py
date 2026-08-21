import contextlib
import os
import json
import re

from dotenv import load_dotenv
from google import genai


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
MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# JSON EXTRACTION
# ============================================================
def extract_json(text):
    """
    Extract JSON from Gemini's response.
    """

    text = text.strip()

    # Remove markdown JSON fences
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

    # Try direct JSON parsing
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(
            text
        )

    # Search for JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:

        raise ValueError(
            "Gemini did not return valid JSON."
        )

    try:

        return json.loads(match[0])

    except json.JSONDecodeError as e:

        raise ValueError(f"Could not parse Gemini JSON: {e}") from e
# CHOOSE ANALYSIS
# ============================================================

def choose_analysis(
    question,
    profile
):
    """
    Ask Gemini to choose the correct
    Python analysis operation.
    """

    prompt = f"""
You are an expert AI Data Analyst.

Your job is to understand the user's question
and choose exactly ONE Python analysis operation.
14. If the question contains a condition such as
"in Delhi", "in Mumbai", "for IT", "for Laptop",
"where City is Delhi", etc., use a filtered operation.

15. For a filtered total, use filtered_sum.

16. For a filtered average, use filtered_average.

17. For a filtered count, use filtered_count.

18. For a filtered unique/distinct count,
use filtered_unique_count.

19. Never invent a filter value.
Use the value provided by the user.

20. Never invent a column name.
Only use columns from the dataset profile.

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

IMPORTANT:

Use this operation when the question contains
words such as:

unique
distinct
different

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

14. filtered_sum

Use when the user asks for a sum/total
with a condition or filter.

Examples:

- What is the total revenue in Delhi?
- What are the sales in Mumbai?
- What is the total salary for the IT department?

Required JSON:

{{
    "operation": "filtered_sum",
    "filter_column": "City",
    "filter_value": "Delhi",
    "value_column": "Revenue"
}}


15. filtered_average

Use when the user asks for an average
with a condition or filter.

Examples:

- What is the average revenue in Delhi?
- What is the average salary in HR?
- What is the average price for laptops?

Required JSON:

{
    "operation": "filtered_average",
    "filter_column": "City",
    "filter_value": "Delhi",
    "value_column": "Revenue"
}


16. filtered_count

Use when the user asks for a count
with a condition or filter.

Examples:

- How many transactions happened in Delhi?
- How many employees are in IT?
- How many orders are from Mumbai?

Required JSON:

{
    "operation": "filtered_count",
    "filter_column": "City",
    "filter_value": "Delhi",
    "count_column": "Invoice_ID"
}


17. filtered_unique_count

Use when the user asks for unique/distinct
values after applying a filter.

Examples:

- How many unique customers are in Delhi?
- How many distinct products were sold in Mumbai?
- How many unique employees are in HR?

Required JSON:

{
    "operation": "filtered_unique_count",
    "filter_column": "City",
    "filter_value": "Delhi",
    "value_column": "Customer_ID"
}
IMPORTANT RULES
===============

1. Use ONLY columns that actually exist.

2. Never invent a column name.

3. Match the user's wording to the
   most appropriate operation.

4. If the user says "unique" or "distinct",
   prefer calculate_unique_count.

5. If the user asks for a total,
   prefer calculate_sum.

6. If the user asks for an average,
   prefer calculate_average.

7. If the user asks for the highest value,
   use calculate_max.

8. If the user asks for the lowest value,
   use calculate_min.

9. If the user asks for "top N",
   use top_n.

10. Return ONLY valid JSON.

11. Do not return markdown.

12. Do not explain your decision.

13. Return exactly ONE JSON object.
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
    Ask Gemini to explain the actual
    Python calculation.
    """

    if hasattr(
        result,
        "to_dict"
    ):

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

2. Use the actual Python result.

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

10. Do not confuse record count with
    unique count.

11. Do not mention internal prompts.

12. Do not mention Gemini.

13. Do not say that more analysis is required
    when the Python result already answers
    the question.

Return only the final natural-language answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip() # pyright: ignore[reportOptionalMemberAccess]