# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import *
from data_loader import load_data_from_gsheet, merge_and_clean
from features import compute_disparities, add_disparity_index
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

# --- Cached Data Loaders ---
@st.cache_data(show_spinner="📥 Loading data from Google Sheets...", persist=True)
def get_merged_data():
    staff_df = load_data_from_gsheet(STAFF_SHEET_ID)
    student_df = load_data_from_gsheet(STUDENT_SHEET_ID)
    merged = merge_and_clean(student_df, staff_df)
    return merged

@st.cache_data(show_spinner="🔍 Computing disparities...", persist=True)
def get_disparity_data(merged):
    merged = compute_disparities(merged, FACULTY_RACE_COLS, STUDENT_RACE_COLS)
    disparity_columns = [f"{race}_disparity" for race in FACULTY_RACE_COLS]
    merged["Disparity Index"] = merged[disparity_columns].abs().mean(axis=1)
    merged = add_disparity_index(merged, FACULTY_RACE_COLS)
    merged["endowment_per_fte"] = merged[
        "Endowment assets (year end) per FTE enrollment (FASB)"
    ].fillna(
        merged["Endowment assets (year end) per FTE enrollment (GASB)"]
    )
    return merged

# --- Load and Process Data ---
merged = get_merged_data()
merged = get_disparity_data(merged)

# --- Sidebar Filters ---
selected_state, selected_control, selected_degree, selected_urban = filter_sidebar(merged)

@st.cache_data(show_spinner="🔍 Filtering dataset...", persist=True)
def get_filtered_df(merged, selected_state, selected_control, selected_degree, selected_urban):
    df = merged.copy()
    if selected_state != "All":
        df = df[df["State abbreviation"] == selected_state]
    if selected_control != "All":
        df = df[df["Public/Private"] == selected_control]
    if selected_degree != "All":
        df = df[df["Institutional category"] == selected_degree]
    if selected_urban != "All":
        df = df[df["Degree of urbanization (Urban-centric locale)"] == selected_urban]
    return df

filtered_df = get_filtered_df(merged, selected_state, selected_control, selected_degree, selected_urban)

# --- Tabs ---
tabs = st.tabs(["📋 Overview", "🧠 PCA & UMAP", "🎯 Disparity Clusters", "📈 Regression"])

# === 📋 Overview ===
with tabs[0]:
    st.subheader("📋 Filtered Institutions with Racial Disparities")

    overview_columns = ["institution name_x"] + [f"{race}_disparity" for race in FACULTY_RACE_COLS] + [
        "State abbreviation", "Public/Private", "Total  enrollment", "Disparity Index"
    ]

    display_df = filtered_df[overview_columns].rename(columns={
        "institution name_x": "Institution",
        "State abbreviation": "State",
        **{f"{race}_disparity": f"{race} disparity" for race in FACULTY_RACE_COLS}
    })

    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    st.markdown("""
    #### ℹ️ How to Interpret Disparity
    - **Disparity = % of faculty − % of students** from the same racial group.
    - A **positive disparity** means the group is **overrepresented among faculty** compared to students.
    - A **negative disparity** means the group is **underrepresented among faculty**.
    - A value close to **zero** suggests relatively proportional representation.

    #### 📌 Note on PCA/UMAP
    Racial disparity variables are **excluded** from the dimensionality reduction (PCA/UMAP) so that clusters represent institutional characteristics — not outcomes.
    This helps ensure that disparities are analyzed as **results**, not drivers of clustering.
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
        "Percent admitted - total",
        "Admissions yield - total",
        "Student-to-faculty ratio",
        "Average salary equated to 9 months of full-time instructional staff - all ranks",
        "Percent of full-time first-time undergraduates awarded any financial aid",
        "Percent of full-time first-time undergraduates awarded federal, state, local or institutional grant aid",
        "Average amount of federal, state, local or institutional grant aid awarded",
        "endowment_per_fte"
    ]

    grad_features = ["Graduation rate, total cohort", "Graduation rate - Bachelor degree within 6 years, total"]

    selected_features = []
    if "Institutional features" in feature_groups:
        selected_features += institutional_features
    if "Graduation rates" in feature_groups:
        selected_features += grad_features

    # Explicitly drop racial disparity columns
    drop_disparity_cols = [f"{race}_disparity" for race in FACULTY_RACE_COLS]
    all_features = selected_features + CATEGORICAL_FEATURES + ["institution name_x"]
    pca_data = filtered_df[all_features].drop(columns=[col for col in drop_disparity_cols if col in filtered_df.columns], errors="ignore").copy()

    column_thresh = 0.7 * len(pca_data)
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

    selected_race_label = st.radio("Choose a race to visualize:", list(disparity_races.keys()), horizontal=True)
    race_column = disparity_races[selected_race_label]

    st.subheader(f"{selected_race_label} Disparity by Cluster")
    plot_df = filtered_df[["institution name_x", race_column]].copy()
    plot_df = plot_df.dropna()
    plot_df = plot_df.merge(pca_data[["institution name_x", "PCA_1", "PCA_2", "UMAP_1", "UMAP_2", "Cluster"]], on="institution name_x", how="inner")

    fig_pca = px.scatter(
        plot_df,
        x="PCA_1", y="PCA_2",
        color=race_column,
        color_continuous_scale=["red", "lightgrey", "blue"],
        range_color=[-abs_max, abs_max],
        hover_data=["institution name_x", "Cluster"],
        title=f"PCA View – {selected_race_label} Disparity"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    fig_umap = px.scatter(
        plot_df,
        x="UMAP_1", y="UMAP_2",
        color=race_column,
        color_continuous_scale=["red", "lightgrey", "blue"],
        range_color=[-abs_max, abs_max],
        hover_data=["institution name_x", "Cluster"],
        title=f"UMAP View – {selected_race_label} Disparity"
    )
    st.plotly_chart(fig_umap, use_container_width=True)

# === 📈 Regression ===
with tabs[3]:
    st.subheader("📈 Linear Regression & Correlation Analysis")

    selected_race = st.selectbox("Choose a race for disparity outcome:", list(FACULTY_RACE_COLS.keys()))
    disparity_var = f"{selected_race}_disparity"

    reg_features = [
        "Total  enrollment",
        "Tuition and fees, 2023-24",
        "Percent admitted - total",
        "Admissions yield - total",
        "Student-to-faculty ratio",
        "Average salary equated to 9 months of full-time instructional staff - all ranks",
        "Percent of full-time first-time undergraduates awarded any financial aid",
        "endowment_per_fte"
    ]

    reg_df = filtered_df[[disparity_var] + reg_features].dropna()

    st.write("### 🔍 Correlation with Disparity")
    corr_matrix = reg_df.corr()[[disparity_var]].drop(index=disparity_var)
    st.dataframe(corr_matrix.style.format("{:.2f}"))

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig)

    st.write("### 🧮 Variance Inflation Factor (VIF)")
    X_vif = sm.add_constant(reg_df[reg_features])
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X_vif.columns
    vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    st.dataframe(vif_df)

    st.write("### 🧮 Linear Regression Results")
    X = reg_df[reg_features]
    y = reg_df[disparity_var]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    st.markdown(model.summary().as_html(), unsafe_allow_html=True)

    st.write("### 📊 Regression Diagnostics")
    predictions = model.predict(X)
    residuals = y - predictions

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Residuals vs. Fitted**")
        fig, ax = plt.subplots()
        ax.scatter(predictions, residuals, alpha=0.5)
        ax.axhline(0, linestyle='--', color='grey')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

    with col2:
        st.write("**Q-Q Plot of Residuals**")
        fig = sm.qqplot(residuals, line='45', fit=True)
        st.pyplot(fig)

    st.write("**Histogram of Residuals**")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Distribution of Residuals")
    st.pyplot(fig)
