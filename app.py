import pandas as pd
import streamlit as st

from data_profile import create_profile
from analyst import run_analysis


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 AI Data Analyst")

st.write(
    "Upload a CSV or Excel dataset and ask questions "
    "using natural language."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # LOAD DATASET
    # ========================================================

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        elif uploaded_file.name.lower().endswith(".xlsx"):

            df = pd.read_excel(
                uploaded_file
            )

        else:

            st.error(
                "Unsupported file type."
            )

            st.stop()

    except Exception as e:

        st.error(
            f"Could not read the file: {e}"
        )

        st.stop()


    # ========================================================
    # SUCCESS MESSAGE
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

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{len(df.columns):,}"
        )

    with col3:

        missing_values = int(
            df.isna()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            f"{missing_values:,}"
        )

    with col4:

        duplicate_rows = int(
            df.duplicated()
            .sum()
        )

        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}"
        )


    # ========================================================
    # DATA PREVIEW
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

        column_info = pd.DataFrame(
            {
                "Column": df.columns,

                "Data Type": [
                    str(df[column].dtype)
                    for column in df.columns
                ],

                "Missing Values": [
                    int(
                        df[column]
                        .isna()
                        .sum()
                    )
                    for column in df.columns
                ],

                "Unique Values": [
                    int(
                        df[column]
                        .nunique()
                    )
                    for column in df.columns
                ],
            }
        )

        st.dataframe(
            column_info,
            use_container_width=True
        )


    # ========================================================
    # AI DATA ANALYST
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Ask the AI Data Analyst"
    )

    st.write(
        "Ask a question about your dataset."
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
    # RUN ANALYSIS
    # ========================================================

    if question.strip():

        # ----------------------------------------------------
        # CREATE DATASET PROFILE
        # ----------------------------------------------------

        with st.spinner(
            "🔍 Understanding your dataset..."
        ):

            try:

                profile = create_profile(
                    df
                )

            except Exception as e:

                st.error(
                    f"Could not profile dataset: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # RUN AI AGENT
        # ----------------------------------------------------

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
        # AI ANSWER
        # ====================================================

        st.subheader(
            "💡 AI Analyst"
        )

        st.write(
            response["explanation"]
        )


        # ====================================================
        # ANALYSIS DETAILS
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


            st.write(
                "**Actual Python result:**"
            )

            result = response["result"]


            # =================================================
            # DATAFRAME RESULT
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
                # AUTOMATIC CHART
                # =============================================

                if len(result.columns) >= 2:

                    chart_data = result.copy()

                    category_column = (
                        chart_data.columns[0]
                    )

                    value_column = (
                        chart_data.columns[1]
                    )


                    # -----------------------------------------
                    # Convert value column to numeric
                    # -----------------------------------------

                    chart_data[value_column] = pd.to_numeric(
                        chart_data[value_column],
                        errors="coerce"
                    )


                    # -----------------------------------------
                    # Remove invalid values
                    # -----------------------------------------

                    chart_data = chart_data.dropna(
                        subset=[
                            value_column
                        ]
                    )


                    # -----------------------------------------
                    # Set category as index
                    # -----------------------------------------

                    chart_data = chart_data.set_index(
                        category_column
                    )


                    # -----------------------------------------
                    # Show chart
                    # -----------------------------------------

                    st.write(
                        "### 📊 Visualization"
                    )

                    st.bar_chart(
                        chart_data[
                            value_column
                        ]
                    )


            # =================================================
            # FLOAT RESULT
            # =================================================

            elif isinstance(
                result,
                float
            ):

                st.write(
                    f"{result:,.2f}"
                )


            # =================================================
            # INTEGER RESULT
            # =================================================

            elif isinstance(
                result,
                int
            ):

                st.write(
                    f"{result:,}"
                )


            # =================================================
            # OTHER RESULT
            # =================================================

            else:

                st.write(
                    result
                )