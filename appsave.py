# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from config import *
from data_loader import load_data_from_gsheet, merge_and_clean
from features import compute_disparities, add_disparity_index
from analysis import run_logistic_regressions
from ui_components import filter_sidebar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap.umap_ as umap

# --- Constants ---
CATEGORICAL_FEATURES = [
    "Public/Private",
    "Degree of urbanization (Urban-centric locale)",
    "Institutional category"
]

# --- App Config ---
st.set_page_config(page_title="Diversity Disparity Dashboard", layout="wide")
st.title("📊 Faculty vs. Student Racial Disparity in U.S. Colleges")

# --- Load Data ---
with st.spinner("📥 Loading data from Google Sheets..."):
    staff_df = load_data_from_gsheet(STAFF_SHEET_ID)
    student_df = load_data_from_gsheet(STUDENT_SHEET_ID)
    merged = merge_and_clean(student_df, staff_df)

# --- Feature Engineering ---
merged = compute_disparities(merged, FACULTY_RACE_COLS, STUDENT_RACE_COLS)
merged = add_disparity_index(merged, FACULTY_RACE_COLS)

# --- Sidebar Filters ---
selected_state, selected_control, selected_degree, selected_urban = filter_sidebar(merged)

filtered_df = merged.copy()
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["State abbreviation"] == selected_state]
if selected_control != "All":
    filtered_df = filtered_df[filtered_df["Public/Private"] == selected_control]
if selected_degree != "All":
    filtered_df = filtered_df[filtered_df["Institutional category"] == selected_degree]
if selected_urban != "All":
    filtered_df = filtered_df[filtered_df["Degree of urbanization (Urban-centric locale)"] == selected_urban]

# --- Tabs ---
tabs = st.tabs(["📋 Overview", "🧠 PCA & UMAP", "🎯 Disparity Clusters", "📈 Regression"])

# === 📋 Overview ===
with tabs[0]:
    st.subheader("📋 Filtered Institutions with Disparity Index")
    st.dataframe(
        filtered_df["institution name_x"].value_counts().reset_index().rename(columns={"index": "Institution", "institution name_x": "Count"}),
        use_container_width=True
    )

    st.markdown("""
    #### ℹ️ How to Interpret Disparity
    - **Disparity = % of faculty − % of students** from the same racial group.
    - A **positive disparity** means the group is **overrepresented among faculty** compared to students.
    - A **negative disparity** means the group is **underrepresented among faculty**.
    - A value close to **zero** suggests relatively proportional representation.
    """)

# === 🧠 PCA & UMAP ===
with tabs[1]:
    st.subheader("🧠 PCA + UMAP + Clustering (Excludes Racial Disparities)")

    feature_groups = st.sidebar.multiselect(
        "Include in clustering:",
        ["Institutional features", "Graduation rates"],
        default=["Institutional features", "Graduation rates"]
    )

    institutional_features = [
        "Total  enrollment",
        "Tuition and fees, 2023-24",
        "Percent admitted - total"
    ]

    grad_features = GRAD_RATE_COLS

    selected_features = []
    if "Institutional features" in feature_groups:
        selected_features += institutional_features
    if "Graduation rates" in feature_groups:
        selected_features += grad_features

    # Prepare Data for PCA/UMAP (drop racial disparities)
    pca_data = filtered_df[selected_features + CATEGORICAL_FEATURES + ["institution name_x"]].copy()
    column_thresh = 0.8 * len(pca_data)
    pca_data = pca_data.dropna(axis=1, thresh=column_thresh)
    pca_data = pca_data.dropna()

    st.caption(f"📊 Institutions included in PCA/UMAP: {pca_data.shape[0]}")

    X_features = pd.get_dummies(pca_data.drop(columns=["institution name_x"]), drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=4)
    pca_model = PCA(n_components=2)
    pca_result = pca_model.fit_transform(X_scaled)
    umap_result = umap.UMAP(random_state=42).fit_transform(X_scaled)
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_scaled)

    pca_data = pca_data.reset_index(drop=True)
    pca_data["PCA_1"] = pca_result[:, 0]
    pca_data["PCA_2"] = pca_result[:, 1]
    pca_data["UMAP_1"] = umap_result[:, 0]
    pca_data["UMAP_2"] = umap_result[:, 1]
    pca_data["Cluster"] = cluster_labels

    st.markdown("""
    #### 🔎 PCA Explained
    - PCA projects data onto axes that explain the most variance.
    - UMAP preserves more local structure and neighborhood proximity.
    - Clustering (KMeans) groups institutions with similar profiles.
    """)

    st.write("##### 🔢 Variance Explained")
    explained_df = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "Variance Explained": pca_model.explained_variance_ratio_
    })
    st.dataframe(explained_df)

    st.write("##### 🔍 Top Features Contributing to PC1/PC2")
    components_df = pd.DataFrame(pca_model.components_, columns=X_features.columns, index=["PC1", "PC2"])
    st.write("**PC1 Top Features:**")
    st.dataframe(components_df.loc["PC1"].sort_values(ascending=False).head(10))
    st.write("**PC2 Top Features:**")
    st.dataframe(components_df.loc["PC2"].sort_values(ascending=False).head(10))

    st.write("### 📌 PCA Scatter Plot")
    fig1 = px.scatter(
        pca_data,
        x="PCA_1", y="PCA_2",
        color="Cluster",
        hover_data=["institution name_x"],
        title="PCA Projection Colored by Cluster"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.write("### 📌 UMAP Scatter Plot")
    fig2 = px.scatter(
        pca_data,
        x="UMAP_1", y="UMAP_2",
        color="Cluster",
        hover_data=["institution name_x"],
        title="UMAP Projection Colored by Cluster"
    )
    st.plotly_chart(fig2, use_container_width=True)

# === 🎯 Disparity Clusters ===
with tabs[2]:
    st.subheader("🎯 Explore Racial Disparity by Cluster")

    disparity_races = {
        "Asian": "Asian_disparity",
        "Black": "Black_disparity",
        "Hispanic": "Hispanic_disparity",
        "White": "White_disparity",
        "Two or more": "Two or more_disparity",
        "Native American": "Native American_disparity",
        "Pacific Islander": "Pacific Islander_disparity"
    }

    all_disparities = filtered_df[[f"{r}_disparity" for r in FACULTY_RACE_COLS]].stack()
    vmin = np.percentile(all_disparities, 1)
    vmax = np.percentile(all_disparities, 99)
    abs_max = max(abs(vmin), abs(vmax))
    if abs_max == 0 or np.isnan(abs_max):
        abs_max = 1

    race_tabs = st.tabs(list(disparity_races.keys()))

    for i, (race_label, race_column) in enumerate(disparity_races.items()):
        with race_tabs[i]:
            st.subheader(f"{race_label} Disparity by Cluster")
            plot_df = filtered_df[["institution name_x", race_column]].copy()
            plot_df = plot_df.dropna()

            # Attach PCA cluster results from previous tab
            plot_df = plot_df.merge(pca_data[["institution name_x", "PCA_1", "PCA_2", "UMAP_1", "UMAP_2", "Cluster"]], on="institution name_x", how="inner")

            fig_pca = px.scatter(
                plot_df,
                x="PCA_1", y="PCA_2",
                color=race_column,
                color_continuous_scale=["red", "lightgrey", "blue"],
                range_color=[-abs_max, abs_max],
                hover_data=["institution name_x", "Cluster"],
                title=f"PCA View – {race_label} Disparity"
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            fig_umap = px.scatter(
                plot_df,
                x="UMAP_1", y="UMAP_2",
                color=race_column,
                color_continuous_scale=["red", "lightgrey", "blue"],
                range_color=[-abs_max, abs_max],
                hover_data=["institution name_x", "Cluster"],
                title=f"UMAP View – {race_label} Disparity"
            )
            st.plotly_chart(fig_umap, use_container_width=True)

# === 📈 Regression ===
with tabs[3]:
    st.subheader("📈 Logistic Regression by Racial Disparity")
    logit_tabs = st.tabs(list(FACULTY_RACE_COLS.keys()))

    for i, race in enumerate(FACULTY_RACE_COLS):
        with logit_tabs[i]:
            disp_var = f"{race}_disparity"
            try:
                result_df = run_logistic_regressions(filtered_df, disp_var, selected_features)
                st.dataframe(result_df.style.format("{:.3f}"))
            except Exception as e:
                st.error(f"Error running regression for {race}: {e}")