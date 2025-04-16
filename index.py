import pandas as pd
import streamlit as st

# --- App Config ---
st.set_page_config(page_title="Diversity Disparity Dashboard", layout="wide")
st.title("📊 Faculty vs. Student Racial Disparity in U.S. Colleges")

# --- Google Sheets Loader ---
def load_data_from_gsheet(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    return pd.read_csv(url)

# --- Sheet IDs ---
STAFF_SHEET_ID = "11TNLxFQSxA7W2bFZIRUSm2t-fDEU6Pz0vA-XEJ6PHbo"
STUDENT_SHEET_ID = "1DI2E5Z9APCrPUd4BXPTgQL-cTP1P6eJb5ASYV7Y4raQ"

# --- Load Data ---
with st.spinner("📥 Loading data..."):
    staff_df = load_data_from_gsheet(STAFF_SHEET_ID)
    student_df = load_data_from_gsheet(STUDENT_SHEET_ID)

# --- Filter to Instructional Staff ---
staff_df = staff_df[
    staff_df["Occupation and full- and part-time status"] == "Instructional staff"
].copy()

# --- Merge on unitid and year ---
merged = pd.merge(student_df, staff_df, on=["unitid", "year"], how="inner")

# --- Filter to Title IV + target institutional categories ---
valid_title_iv = "Title IV postsecondary institution"
valid_categories = [
    "Degree-granting, primarily baccalaureate or above",
    "Degree-granting, associate's and certificates",
    "Degree-granting, not primarily baccalaureate or above",
    "Degree-granting, graduate with no undergraduate degrees"
]

filtered = merged[
    (merged["Postsecondary and Title IV institution indicator"] == valid_title_iv) &
    (merged["Institutional category"].isin(valid_categories))
].copy()

# --- Define mappings of race categories ---
faculty_race_cols = {
    "Asian": "Asian total",
    "Black": "Black or African American total",
    "Hispanic": "Hispanic or Latino total",
    "White": "White total",
    "Two or more": "Two or more races total",
    "Native American": "American Indian or Alaska Native total",
    "Pacific Islander": "Native Hawaiian or Other Pacific Islander total"
}

student_race_cols = {
    "Asian": "Percent of total enrollment that are Asian",
    "Black": "Percent of total enrollment that are Black or African American",
    "Hispanic": "Percent of total enrollment that are Hispanic/Latino",
    "White": "Percent of total enrollment that are White",
    "Two or more": "Percent of total enrollment that are two or more races",
    "Native American": "Percent of total enrollment that are American Indian or Alaska Native",
    "Pacific Islander": "Percent of total enrollment that are Native Hawaiian or Other Pacific Islander"
}

# --- Calculate disparities (faculty % - student %) ---
filtered["faculty_total"] = filtered["Grand total"]

for race in faculty_race_cols:
    filtered[f"{race}_faculty_pct"] = (
        filtered[faculty_race_cols[race]] / filtered["faculty_total"]
    ) * 100
    filtered[f"{race}_student_pct"] = filtered[student_race_cols[race]]
    filtered[f"{race}_disparity"] = (
        filtered[f"{race}_faculty_pct"] - filtered[f"{race}_student_pct"]
    )

# --- Streamlit UI ---
st.subheader("🧮 Disparity Viewer")
selected_race = st.selectbox("Select a racial group to view disparities", list(faculty_race_cols.keys()))

# --- Display table ---
display_df = filtered[[
    "institution name_x",
    "year",
    f"{selected_race}_faculty_pct",
    f"{selected_race}_student_pct",
    f"{selected_race}_disparity"
]].rename(columns={
    "institution name_x": "Institution",
    f"{selected_race}_faculty_pct": "Faculty %",
    f"{selected_race}_student_pct": "Student %",
    f"{selected_race}_disparity": "Disparity"
})

st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

st.markdown("""
#### ℹ️ Interpretation:
- **Disparity = Faculty % − Student %**
- Positive values = Overrepresentation in faculty
- Negative values = Underrepresentation in faculty
""")


