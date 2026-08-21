import pandas as pd


def create_profile(df):
    """
    Create a lightweight description of the dataset
    for Gemini.
    """

    profile = {"rows": len(df), "columns": len(df.columns), "column_details": []}

    for column in df.columns:

        series = df[column]

        sample_values = (
            series
            .dropna()
            .head(5)
            .tolist()
        )

        profile["column_details"].append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique()),
                "sample_values": [
                    str(value)
                    for value in sample_values
                ],
            }
        )

    return profile