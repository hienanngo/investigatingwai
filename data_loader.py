# data_loader.py
import pandas as pd

def load_data_from_gsheet(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    return pd.read_csv(url)

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
