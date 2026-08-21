import pandas as pd


def create_profile(df):

    profile = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [],
    }


    for column in df.columns:

        series = df[column]

        info = {
            "name": column,
            "dtype": str(series.dtype),
            "missing": int(
                series.isna().sum()
            ),
            "unique_values": int(
                series.nunique(
                    dropna=True
                )
            ),
        }


        # ====================================================
        # NUMERIC INFORMATION
        # ====================================================

        if pd.api.types.is_numeric_dtype(
            series
        ):

            info["type"] = "numeric"

            info["min"] = (
                float(series.min())
                if series.notna().any()
                else None
            )

            info["max"] = (
                float(series.max())
                if series.notna().any()
                else None
            )

            info["mean"] = (
                float(series.mean())
                if series.notna().any()
                else None
            )


        # ====================================================
        # DATETIME INFORMATION
        # ====================================================

        elif (
            pd.api.types.is_datetime64_any_dtype(
                series
            )
        ):

            info["type"] = "datetime"


        else:

            # Try detecting dates stored as strings
            converted = pd.to_datetime(
                series,
                errors="coerce"
            )

            date_ratio = (
                converted.notna().mean()
                if len(series) > 0
                else 0
            )

            if date_ratio > 0.8:

                info["type"] = "datetime"

            else:

                info["type"] = "categorical"


                # Give Gemini examples
                values = (
                    series
                    .dropna()
                    .astype(str)
                    .unique()
                    [:10]
                )

                info["sample_values"] = (
                    values.tolist()
                )


        profile["columns"].append(
            info
        )


    return profile