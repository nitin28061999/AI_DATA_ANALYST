import pandas as pd
import streamlit as st

from data_profile import create_profile
from analyst import run_analysis # pyright: ignore[reportAttributeAccessIssue]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 AI Data Analyst"
)

st.write(
    "Upload a CSV or Excel dataset and ask "
    "questions using natural language."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx"
    ],
)


# ============================================================
# APPLICATION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:

        if uploaded_file.name.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error(
            f"Could not read dataset: {e}"
        )

        st.stop()


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        f"Loaded {len(df):,} rows × "
        f"{len(df.columns):,} columns"
    )


    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader(
        "📋 Dataset Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with c2:

        st.metric(
            "Columns",
            f"{len(df.columns):,}"
        )

    with c3:

        missing = int(
            df.isna()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            f"{missing:,}"
        )

    with c4:

        duplicates = int(
            df.duplicated()
            .sum()
        )

        st.metric(
            "Duplicate Rows",
            f"{duplicates:,}"
        )


    # ========================================================
    # PREVIEW
    # ========================================================

    with st.expander(
        "👀 Preview dataset"
    ):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    with st.expander(
        "🔎 Column information"
    ):

        column_info = pd.DataFrame({

            "Column": df.columns,

            "Data Type": [
                str(
                    df[c].dtype
                )
                for c in df.columns
            ],

            "Missing Values": [
                int(
                    df[c].isna().sum()
                )
                for c in df.columns
            ],

            "Unique Values": [
                int(
                    df[c].nunique()
                )
                for c in df.columns
            ],

        })

        st.dataframe(
            column_info,
            use_container_width=True
        )


    # ========================================================
    # AI ANALYST
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Ask the AI Data Analyst"
    )

    st.write(
        "Ask a question about your dataset."
    )


    # ========================================================
    # EXAMPLE QUESTIONS
    # ========================================================

    st.info(
        """
Try questions like:

• What is the total revenue?

• What are the top 5 cities by revenue?

• Which brand has the highest revenue?

• What is the average selling price?

• What percentage of revenue comes from each city?

• What is the monthly revenue?

• How many customers are there?

• What is the average salary by department?
"""
    )


    # ========================================================
    # QUESTION
    # ========================================================

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What is the total revenue?"
        ),
    )


    # ========================================================
    # RUN
    # ========================================================

    if question.strip():

        # ====================================================
        # PROFILE
        # ====================================================

        with st.spinner(
            "🔍 Understanding your dataset..."
        ):

            try:

                profile = create_profile(
                    df
                )

            except Exception as e:

                st.error(
                    f"Profiling failed: {e}"
                )

                st.stop()


        # ====================================================
        # ANALYSIS
        # ====================================================

        with st.spinner(
            "🤖 Analyzing your question..."
        ):

            try:

                response = run_analysis(
                    df,
                    profile,
                    question
                )

            except Exception as e:

                st.error(
                    f"Analysis failed: {e}"
                )

                st.stop()


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader(
            "💡 AI Analyst"
        )

        st.write(
            response["explanation"]
        )


        # ====================================================
        # DETAILS
        # ====================================================

        with st.expander(
            "🔧 Analysis details"
        ):

            st.write(
                "**Operation selected by AI:**"
            )

            st.json(
                response["plan"]
            )


            result = response["result"]

            st.write(
                "**Actual Python result:**"
            )


            # =================================================
            # DATAFRAME
            # =================================================

            if isinstance(
                result,
                pd.DataFrame
            ):

                st.dataframe(
                    result,
                    use_container_width=True
                )


                # =============================================
                # VISUALIZATION
                # =============================================

                if (
                    len(result.columns) >= 2
                    and len(result) > 0
                ):

                    st.subheader(
                        "📊 Visualization"
                    )

                    chart_data = result.copy()

                    category_column = (
                        chart_data.columns[0]
                    )

                    value_column = (
                        chart_data.columns[1]
                    )


                    try:

                        chart_data = (
                            chart_data
                            .set_index(
                                category_column
                            )
                        )

                        st.bar_chart(
                            chart_data[
                                value_column
                            ]
                        )

                    except Exception:

                        st.info(
                            "A chart could not be generated "
                            "for this result."
                        )


            # =================================================
            # NUMBERS
            # =================================================

            elif isinstance(
                result,
                (float, int)
            ):

                st.write(
                    f"{result:,.2f}"
                    if isinstance(
                        result,
                        float
                    )
                    else
                    f"{result:,}"
                )


            # =================================================
            # OTHER
            # =================================================

            else:

                st.write(
                    result
                )