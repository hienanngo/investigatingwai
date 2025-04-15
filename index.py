
import streamlit as st

# --- Page Setup ---
st.set_page_config(page_title="Faculty Diversity Dashboard", layout="wide")
st.title("🎓 Faculty vs. Student Diversity in U.S. Colleges")

import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- Load from Google Sheets ---
def load_data_from_gsheet(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    return pd.read_csv(url)

# --- Google Sheet IDs ---
STAFF_SHEET_ID = "11TNLxFQSxA7W2bFZIRUSm2t-fDEU6Pz0vA-XEJ6PHbo"
STUDENT_SHEET_ID = "1DI2E5Z9APCrPUd4BXPTgQL-cTP1P6eJb5ASYV7Y4raQ"

with st.spinner("📥 Loading data from Google Sheets..."):
    staff_df = load_data_from_gsheet(STAFF_SHEET_ID)
    student_df = load_data_from_gsheet(STUDENT_SHEET_ID)

# --- Filter to Instructional Staff ---
faculty_filtered = staff_df[
    staff_df["S2023_OC.Occupation and full- and part-time status"] == "Instructional staff"
].copy()
faculty_filtered["total_faculty"] = faculty_filtered["S2023_OC.Grand total"]

# --- Race Columns ---
faculty_race_cols = {
    "Asian": "S2023_OC.Asian total",
    "Black": "S2023_OC.Black or African American total",
    "Hispanic": "S2023_OC.Hispanic or Latino total",
    "White": "S2023_OC.White total",
    "Two or more": "S2023_OC.Two or more races total",
    "Native American": "S2023_OC.American Indian or Alaska Native total",
    "Pacific Islander": "S2023_OC.Native Hawaiian or Other Pacific Islander total"
}

student_race_cols = {
    "Asian": "DRVEF2023.Percent of total enrollment that are Asian",
    "Black": "DRVEF2023.Percent of total enrollment that are Black or African American",
    "Hispanic": "DRVEF2023.Percent of total enrollment that are Hispanic/Latino",
    "White": "DRVEF2023.Percent of total enrollment that are White",
    "Two or more": "DRVEF2023.Percent of total enrollment that are two or more races",
    "Native American": "DRVEF2023.Percent of total enrollment that are American Indian or Alaska Native",
    "Pacific Islander": "DRVEF2023.Percent of total enrollment that are Native Hawaiian or Other Pacific Islander"
}

# --- Normalize faculty race counts to percentages ---
for race, col in faculty_race_cols.items():
    faculty_filtered[f"{race}_faculty_pct"] = (faculty_filtered[col] / faculty_filtered["total_faculty"]) * 100

# --- Keep only necessary columns for merge ---
faculty_subset = faculty_filtered[["unitid", "year"] + [f"{race}_faculty_pct" for race in faculty_race_cols]]

# --- Prepare student data ---
student_subset = student_df[["unitid", "institution name", "year", 
                             "HD2023.Postsecondary and Title IV institution indicator", 
                             "HD2023.Institutional category"] + list(student_race_cols.values())].copy()
student_subset.rename(columns={v: f"{k}_student_pct" for k, v in student_race_cols.items()}, inplace=True)

# --- Merge datasets ---
merged = pd.merge(student_subset, faculty_subset, on=["unitid", "year"], how="inner")

# --- Filter by Title IV and degree-granting institutions ---
valid_categories = [
    "Degree-granting, primarily baccalaureate or above",
    "Degree-granting, associate's and certificates",
    "Degree-granting, not primarily baccalaureate or above",
    "Degree-granting, graduate with no undergraduate degrees"
]
merged = merged[
    (merged["HD2023.Postsecondary and Title IV institution indicator"] == "Title IV postsecondary institution") &
    (merged["HD2023.Institutional category"].isin(valid_categories))
]

# --- Calculate ratios ---
for race in faculty_race_cols.keys():
    merged[f"{race}_ratio"] = merged[f"{race}_faculty_pct"] / merged[f"{race}_student_pct"]
    merged[f"{race}_ratio"] = merged[f"{race}_ratio"].replace([float('inf'), -float('inf')], pd.NA)

merged.fillna(0, inplace=True)

# --- UI: Race selection ---
st.subheader("📊 Explore Representation Ratios")
selected_race = st.selectbox("Choose a racial group to analyze", list(faculty_race_cols.keys()))

# --- Display table ---
st.dataframe(
    merged[["institution name", f"{selected_race}_faculty_pct", f"{selected_race}_student_pct", f"{selected_race}_ratio"]]
    .sort_values(by=f"{selected_race}_ratio", ascending=True)
    .rename(columns={
        f"{selected_race}_faculty_pct": "Faculty %",
        f"{selected_race}_student_pct": "Student %",
        f"{selected_race}_ratio": "Faculty-to-Student Ratio"
    })
    .reset_index(drop=True)
)

st.markdown("""
#### ℹ️ Interpretation:
- **Ratio = 1.0** → Representation is equal.
- **< 1.0** → Underrepresented in faculty.
- **> 1.0** → Overrepresented in faculty.
""")