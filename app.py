import pandas as pd
import streamlit as st

from analyst import analyze_data # pyright: ignore[reportAttributeAccessIssue]
from data_profile import profile_data

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")
st.markdown("Upload a CSV file and ask questions about your dataset in natural language.")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

        # Tabs for interface navigation
        tab_analysis, tab_profile, tab_preview = st.tabs(["💬 Ask AI Analyst", "📈 Data Profile", "📋 Raw Data"])

        with tab_preview:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(50), use_container_width=True)

        with tab_profile:
            st.subheader("Dataset Metadata & Profile")
            profile = profile_data(df)
            st.json(profile)

        with tab_analysis:
            st.subheader("Query Your Data")
            question = st.text_input("Enter your question:", placeholder="e.g., What is the total revenue for Delhi?")

            if st.button("Run Analysis", type="primary"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Analyzing dataset with AI..."):
                        try:
                            output = analyze_data(df, question)

                            st.markdown("### Answer")
                            st.write(output["explanation"])

                            with st.expander("Show Execution Details & Execution Plan"):
                                st.markdown("**Generated Plan:**")
                                st.json(output["plan"])
                                st.markdown("**Raw Execution Result:**")
                                st.write(output["result"])

                        except Exception as e:
                            st.error(f"Analysis Error: {str(e)}")

    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
else:
    st.info("Please upload a CSV file to get started.")