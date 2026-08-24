from __future__ import annotations

from typing import Any, Callable
import pandas as pd
import streamlit as st

from data_profile import create_profile
import analyst  # imported module directly to avoid Pyright attribute resolution issues

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Data Analyst")
st.write("Upload a CSV or Excel dataset and ask questions using natural language.")

# ============================================================
# FILE UPLOAD & DATA LOADING
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df: pd.DataFrame = pd.read_csv(uploaded_file)
        else:
            df: pd.DataFrame = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read dataset: {e}")
        st.stop()

    st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")

    # ============================================================
    # DATASET OVERVIEW METRICS
    # ============================================================

    st.subheader("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Rows", f"{len(df):,}")

    with c2:
        st.metric("Columns", f"{len(df.columns):,}")

    with c3:
        missing = int(df.isna().sum().sum())
        st.metric("Missing Values", f"{missing:,}")

    with c4:
        duplicates = int(df.duplicated().sum())
        st.metric("Duplicate Rows", f"{duplicates:,}")

    # ============================================================
    # DATASET INSPECTION
    # ============================================================

    with st.expander("👀 Preview dataset"):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("🔎 Column information"):
        column_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [str(df[c].dtype) for c in df.columns],
                "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
                "Unique Values": [int(df[c].nunique()) for c in df.columns],
            }
        )
        st.dataframe(column_info, use_container_width=True)

    # ============================================================
    # AI ANALYST SECTION
    # ============================================================

    st.divider()
    st.subheader("🤖 Ask the AI Data Analyst")

    st.info(
        """
        **Try asking questions like:**
        * What is the total revenue?
        * What are the top 5 cities by revenue?
        * Which brand has the highest sales?
        * What is the percentage share of revenue by region?
        * What is the monthly breakdown of total sales?
        """
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: What is the total revenue?",
    )

    if question.strip():
        # Profile Data
        with st.spinner("🔍 Understanding your dataset..."):
            try:
                profile = create_profile(df)
            except Exception as e:
                st.error(f"Profiling failed: {e}")
                st.stop()

        # Run Analysis
        with st.spinner("🤖 Analyzing your question..."):
            try:
                # Resolve run_analysis dynamically or from module to avoid pyright attributes issue
                run_analysis_fn: Callable[..., dict[str, Any]] = getattr(
                    analyst, "run_analysis"
                )
                response = run_analysis_fn(df, profile, question)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # Render Response
        st.subheader("💡 AI Analyst")
        st.write(response.get("explanation", "Analysis completed."))

        # Technical Execution Details
        with st.expander("🔧 Analysis details"):
            st.write("**Operation selected by AI:**")
            st.json(response.get("plan", {}))

            result = response.get("result")
            st.write("**Actual Python result:**")

            # DataFrame Output
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)

                if len(result.columns) >= 2 and len(result) > 0:
                    st.subheader("📊 Visualization")
                    try:
                        chart_data = result.copy()
                        category_column = chart_data.columns[0]
                        value_column = chart_data.columns[1]

                        chart_data = chart_data.set_index(category_column)
                        st.bar_chart(chart_data[value_column])
                    except Exception:
                        st.info("A chart could not be generated for this result.")

            # Numeric Output
            elif isinstance(result, (float, int)):
                formatted_num = (
                    f"{result:,.2f}"
                    if isinstance(result, float)
                    else f"{result:,}"
                )
                st.subheader(formatted_num)

            # Generic Output
            else:
                st.write(result)