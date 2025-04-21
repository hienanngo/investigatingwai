# data_loader.py
import pandas as pd

# --- Google Sheets Loader ---
def load_data_from_gsheet(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()  # ← this is the key line to normalize column names
    return df


# --- Google Sheet IDs ---
STAFF_SHEET_ID = "11TNLxFQSxA7W2bFZIRUSm2t-fDEU6Pz0vA-XEJ6PHbo"
STUDENT_SHEET_ID = "1DI2E5Z9APCrPUd4BXPTgQL-cTP1P6eJb5ASYV7Y4raQ"


def merge_and_clean(student_df, staff_df):
    staff_df = staff_df[
        staff_df["Occupation and full- and part-time status"] == "Instructional staff"
    ]
    merged = pd.merge(student_df, staff_df, on=["unitid", "year"], how="inner")

    valid_categories = [
        "Degree-granting, primarily baccalaureate or above",
        "Degree-granting, associate's and certificates",
        "Degree-granting, not primarily baccalaureate or above",
        "Degree-granting, graduate with no undergraduate degrees"
    ]

    merged = merged[
        (merged["Postsecondary and Title IV institution indicator"] == "Title IV postsecondary institution") &
        (merged["Institutional category"].isin(valid_categories)) &
        (~merged["Degree of urbanization (Urban-centric locale)"].isin(["{Not available}"])) &
        (~merged["Institution size category"].isin(["Not reported", "Not applicable"]))
    ]

    merged["Public/Private"] = merged["Control of institution"].map({
        "Public": "Public",
        "Private for-profit": "Private for-profit",
        "Private not-for-profit": "Private not-for-profit"
    })

    return merged
