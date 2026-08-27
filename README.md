# AI Data Analyst

A Streamlit app that lets you upload a CSV and ask questions about it in plain
English. Gemini plans which analysis to run (sum, average, group-by, filter,
top-N, etc.); Python (pandas) is the only thing that ever touches the actual
numbers, so the AI can't "hallucinate" a result — it can only choose and
explain a calculation that real code executed.
