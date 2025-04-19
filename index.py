import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from sklearn.cluster import KMeans
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Faculty Diversity Dashboard", layout="wide")
st.title("🎓 Faculty vs. Student Diversity in U.S. Colleges")

import pandas as pd
import numpy as np
import logging

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
    staff_df["Occupation and full- and part-time status"] == "Instructional staff"
].copy()
faculty_filtered["total_faculty"] = faculty_filtered["Grand total"]

# --- Race Columns ---
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

# --- Normalize faculty race counts to percentages ---
for race, col in faculty_race_cols.items():
    faculty_filtered[f"{race}_faculty_pct"] = (faculty_filtered[col] / faculty_filtered["total_faculty"]) * 100

# --- Keep only necessary columns for merge ---
faculty_subset = faculty_filtered[["unitid", "year"] + [f"{race}_faculty_pct" for race in faculty_race_cols]]

# --- Prepare student data ---
student_subset = student_df[["unitid", "institution name", "year", 
                             "Postsecondary and Title IV institution indicator", 
                             "Institutional category"] + list(student_race_cols.values())].copy()
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
    (merged["Postsecondary and Title IV institution indicator"] == "Title IV postsecondary institution") &
    (merged["Institutional category"].isin(valid_categories))
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

st.markdown("---")
st.header("📊 Advanced Analysis")

# --- Data prep for advanced analyses ---
faculty_pct_cols = [f"{race}_faculty_pct" for race in faculty_race_cols]
X = merged[faculty_pct_cols].fillna(0)
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
merged["Cluster"] = clusters

# --- Correlation Matrix ---
st.subheader("📈 Correlation Between Faculty Race Percentages")
corr = merged[faculty_pct_cols].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)
st.markdown("This heatmap shows how race representation metrics are correlated among faculty members across institutions.")

st.subheader("🔍 Average Faculty Diversity by Cluster")
cluster_means = merged.groupby("Cluster")[[f"{race}_faculty_pct" for race in faculty_race_cols]].mean()
st.dataframe(cluster_means.style.format("{:.2f}"))

# --- PCA ---
st.subheader("🧭 PCA of Faculty Representation")

# Run PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Institution"] = merged["institution name"]
pca_df["Cluster"] = merged["Cluster"]

# Interactive PCA plot
fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color=pca_df["Cluster"].astype(str),
    hover_name="Institution",
    title="PCA of Faculty Diversity by Institution Cluster",
    labels={"Cluster": "Cluster"},
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("PCA reduces faculty race data into two dimensions, showing groupings of institutions based on diversity composition. Clusters highlight similar diversity profiles.")

# --- UMAP ---
st.subheader("🌐 UMAP of Faculty Representation")

# Run UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_result = umap_model.fit_transform(X_scaled)
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
umap_df["Institution"] = merged["institution name"]
umap_df["Cluster"] = merged["Cluster"]

# Interactive UMAP plot
fig = px.scatter(
    umap_df,
    x="UMAP1",
    y="UMAP2",
    color=umap_df["Cluster"].astype(str),
    hover_name="Institution",
    title="UMAP Projection of Faculty Race Composition",
    labels={"Cluster": "Cluster"},
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("UMAP reveals nuanced local relationships in the diversity data, preserving structure among institutions without text clutter.")

# --- Clustering ---
st.subheader("🧩 Clustering Institutions by Faculty Composition")
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
merged["Cluster"] = clusters
pca_df["Cluster"] = clusters

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=80)
st.pyplot(fig)
st.markdown("This KMeans clustering shows how institutions group based on similarities in their faculty diversity makeup.")