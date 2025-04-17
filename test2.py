import pandas as pd
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap.umap_ as umap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# --- App Config ---
st.set_page_config(page_title="Diversity Disparity Dashboard", layout="wide")
st.title("📊 Faculty vs. Student Racial Disparity in U.S. Colleges")

# --- Google Sheets Loader ---
def load_data_from_gsheet(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    return pd.read_csv(url)

# --- Google Sheet IDs ---
STAFF_SHEET_ID = "11TNLxFQSxA7W2bFZIRUSm2t-fDEU6Pz0vA-XEJ6PHbo"
STUDENT_SHEET_ID = "1DI2E5Z9APCrPUd4BXPTgQL-cTP1P6eJb5ASYV7Y4raQ"

# --- Load Data ---
with st.spinner("📥 Loading data from Google Sheets..."):
    staff_df = load_data_from_gsheet(STAFF_SHEET_ID)
    student_df = load_data_from_gsheet(STUDENT_SHEET_ID)

# --- Filter instructional staff only ---
staff_df = staff_df[
    staff_df["Occupation and full- and part-time status"] == "Instructional staff"
].copy()

# --- Merge datasets on unitid and year ---
merged = pd.merge(
    student_df, staff_df, on=["unitid", "year"], how="inner", suffixes=("_student", "_staff")
)

# --- Clean filtering ---
merged = merged[
    (merged["Postsecondary and Title IV institution indicator"] == "Title IV postsecondary institution") &
    (merged["Institutional category"].isin([
        "Degree-granting, primarily baccalaureate or above",
        "Degree-granting, associate's and certificates",
        "Degree-granting, not primarily baccalaureate or above",
        "Degree-granting, graduate with no undergraduate degrees"
    ])) &
    (~merged["Degree of urbanization (Urban-centric locale)"].isin(["{Not available}"]))
].copy()

# Drop duplicate column
merged = merged.loc[:, ~merged.columns.duplicated()]

# --- Public/Private Mapping ---
control_map = {
    "Public": "Public",
    "Private for-profit": "Private for-profit",
    "Private not-for-profit": "Private not-for-profit"
}
merged["Public/Private"] = merged["Control of institution"].map(control_map)

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

# --- Calculate Disparities ---
merged["faculty_total"] = merged["Grand total"]
for race in faculty_race_cols:
    merged[f"{race}_faculty_pct"] = (merged[faculty_race_cols[race]] / merged["faculty_total"]) * 100
    merged[f"{race}_student_pct"] = merged[student_race_cols[race]]
    merged[f"{race}_disparity"] = merged[f"{race}_faculty_pct"] - merged[f"{race}_student_pct"]

# --- Graduation rate columns ---
grad_rate_cols = [
    "Graduation rate, American Indian or Alaska Native",
    "Graduation rate, Asian/Native Hawaiian/Other Pacific Islander",
    "Graduation rate, Asian",
    "Graduation rate, Native Hawaiian or Other Pacific Islander",
    "Graduation rate, Black, non-Hispanic",
    "Graduation rate, Hispanic",
    "Graduation rate, White, non-Hispanic",
    "Graduation rate, two or more races"
]

# --- Select data for PCA and clustering ---
categorical_features = [
    "Public/Private",
    "Degree of urbanization (Urban-centric locale)",
    "Institutional category"
]

numerical_features = [
    "Total  enrollment",
    "Tuition and fees, 2023-24",
    "Percent admitted - total",
    "Graduation rate, total cohort"
] + grad_rate_cols + [f"{r}_disparity" for r in faculty_race_cols]

# --- Drop rows with excessive missing data ---
cluster_df = merged[numerical_features + categorical_features + ["institution name_student"]].dropna()
cluster_df_encoded = pd.get_dummies(cluster_df, columns=categorical_features, drop_first=True)

# --- Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df_encoded.drop(columns=["institution name_student"]))

# --- Clustering Controls ---
st.sidebar.header("⚙️ Clustering Controls")
num_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=5)

# --- Compute Embeddings (cached) ---
@st.cache_data(show_spinner="🔍 Computing PCA, UMAP, and Clustering...")
def compute_embeddings(X, k):
    pca_result = PCA(n_components=2).fit_transform(X)
    umap_result = umap.UMAP(random_state=42).fit_transform(X)
    cluster_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    return pca_result, umap_result, cluster_labels

pca_result, umap_result, cluster_labels = compute_embeddings(X_scaled, num_clusters)

# --- Attach results back ---
cluster_df_encoded["PCA_1"] = pca_result[:, 0]
cluster_df_encoded["PCA_2"] = pca_result[:, 1]
cluster_df_encoded["UMAP_1"] = umap_result[:, 0]
cluster_df_encoded["UMAP_2"] = umap_result[:, 1]
cluster_df_encoded["Cluster"] = cluster_labels
cluster_df_encoded["Institution"] = cluster_df["institution name_student"].values

# --- Show Data Summary ---
st.subheader("📊 Cluster Sizes")
st.dataframe(cluster_df_encoded["Cluster"].value_counts().reset_index().rename(columns={"index": "Cluster", "Cluster": "Count"}))

# --- PCA/UMAP Plots ---
st.subheader("🗺️ Cluster Visualization")
fig_pca = px.scatter(
    cluster_df_encoded,
    x="PCA_1", y="PCA_2",
    color=cluster_df_encoded["Cluster"].astype(str),
    hover_data=["Institution"],
    title="PCA Projection"
)
st.plotly_chart(fig_pca, use_container_width=True)

fig_umap = px.scatter(
    cluster_df_encoded,
    x="UMAP_1", y="UMAP_2",
    color=cluster_df_encoded["Cluster"].astype(str),
    hover_data=["Institution"],
    title="UMAP Projection"
)
st.plotly_chart(fig_umap, use_container_width=True)