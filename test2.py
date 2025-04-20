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
st.text("In this dashboard, a cluster represents a group of colleges or universities that share similar characteristics based on various metrics such as racial disparities between faculty and students, graduation rates by race, total enrollment, tuition costs, admission rates, and institutional type. Using the K-Means clustering algorithm, schools are grouped together based on patterns in this data, with each cluster reflecting a unique profile of institutional behavior and demographic makeup. The clusters are visualized using PCA and UMAP, which reduce the high-dimensional data into two dimensions for easy interpretation—schools positioned close together in the plots are more similar. Interpreting these clusters allows users to identify meaningful patterns, such as which types of institutions tend to have higher or lower racial disparities, or how factors like being public or private relate to faculty diversity.")
cluster_counts = cluster_df_encoded["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]
st.dataframe(cluster_counts)

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

st.text("Principal Component Analysis (PCA) reduces the dimensionality of the data by identifying the directions (principal components) of maximum variance. The first two principal components are visualized on a 2D plot, showing how institutions differ based on features like enrollment, tuition, and racial disparities. Institutions closer together in the PCA plot share similar characteristics, while those farther apart have distinct profiles, such as differing levels of faculty diversity or graduation rates.")

fig_umap = px.scatter(
    cluster_df_encoded,
    x="UMAP_1", y="UMAP_2",
    color=cluster_df_encoded["Cluster"].astype(str),
    hover_data=["Institution"],
    title="UMAP Projection"
)
st.plotly_chart(fig_umap, use_container_width=True)

st.text("UMAP (Uniform Manifold Approximation and Projection) also reduces dimensionality but focuses on preserving both local and global relationships in the data. It creates a 2D map where similar institutions are grouped together, revealing more complex patterns that may not be captured by PCA. UMAP is useful for uncovering subtle clusters based on faculty-student racial disparities and other institutional features.")

# st.write(cluster_df_encoded.columns)

# --- Correlation Matrix of Disparities vs. Graduation Rates ---
st.subheader("📈 Correlation Matrix of Disparities vs. Graduation Rates")

# Prepare the columns for the correlation matrix
# First, prepare the list of disparities and graduation rates
disparity_columns = [f"{race}_disparity" for race in faculty_race_cols]
grad_rate_columns = [
    "Graduation rate, Black, non-Hispanic", 
    "Graduation rate, White, non-Hispanic", 
    "Graduation rate, two or more races", 
    "Graduation rate, American Indian or Alaska Native",
    "Graduation rate, Native Hawaiian or Other Pacific Islander"
]

# Extract the relevant columns from the DataFrame for correlation
grad_rate_disparity_columns = disparity_columns + grad_rate_columns
numeric_cols = cluster_df_encoded[grad_rate_disparity_columns]

# Compute the correlation matrix
corr_matrix = numeric_cols.corr()

# Only select the part of the matrix where disparities are on the x-axis and graduation rates on the y-axis
corr_matrix_selected = corr_matrix.loc[grad_rate_columns, disparity_columns]

# Plot the correlation matrix using Seaborn
fig_corr, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(corr_matrix_selected, annot=True, cmap="YlGnBu", center=0, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix of Disparities vs. Graduation Rates", fontsize=16)
ax.set_xlabel('Faculty-Student Disparity')
ax.set_ylabel('Graduation Rate')

st.pyplot(fig_corr)

# Explanation Text
st.text("The correlation matrix above shows the relationships between racial disparities in faculty-student composition and graduation rates for each racial group. Each row corresponds to the graduation rate for a specific racial group, and each column represents the disparity in faculty composition for that group. Positive correlations indicate that greater disparities in faculty diversity are associated with higher graduation rates, while negative correlations suggest the opposite. This matrix helps understand how faculty-student diversity disparities may influence academic success rates across different racial and ethnic groups.")
