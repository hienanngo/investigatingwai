# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.figure_factory as ff
from scipy.stats import chi2_contingency


from config import *
from data_loader import load_data_from_gsheet, merge_and_clean
from features import compute_disparities, add_disparity_index
from ui_components import filter_sidebar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap.umap_ as umap
import plotly.figure_factory as ff
import scipy.stats as ss

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
tabs = st.tabs(["📋 Overview", "🧠 PCA & UMAP", "🎯 Disparity Clusters", "📈 Regression", "📊 Correlation Matrix"])

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

    selected_features = institutional_features + grad_features


    # Explicitly drop racial disparity columns
    drop_disparity_cols = [f"{race}_disparity" for race in FACULTY_RACE_COLS]
    all_features = selected_features + CATEGORICAL_FEATURES + ["institution name_x"]
    pca_data = filtered_df[all_features].drop(columns=[col for col in drop_disparity_cols if col in filtered_df.columns], errors="ignore").copy()

    column_thresh = 0.7 * len(pca_data)
    pca_data = pca_data.dropna(axis=1, thresh=column_thresh)
    pca_data = pca_data.replace([np.inf, -np.inf], np.nan)
    pca_data = pca_data.dropna()

    st.caption(f"📊 Institutions included in PCA/UMAP: {pca_data.shape[0]}")

    X_features = pd.get_dummies(pca_data.drop(columns=["institution name_x"]), drop_first=True)
    X_features = X_features.select_dtypes(include=[np.number])  # Only keep numeric
    X_features = X_features.replace([np.inf, -np.inf], np.nan).dropna()  # Drop inf/nan
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

    reg_df = filtered_df[[disparity_var] + reg_features + ["Public/Private"]].dropna()
    reg_df = pd.get_dummies(reg_df, columns=["Public/Private"], drop_first=True)

    public_private_dummies = [col for col in reg_df.columns if col.startswith("Public/Private_")]
    reg_features += public_private_dummies

    st.write("### 🔍 Correlation with Disparity")
    corr_matrix = reg_df.corr()[[disparity_var]].drop(index=disparity_var)
    st.dataframe(corr_matrix.style.format("{:.2f}"))

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig)

# === 🧮 Variance Inflation Factor (VIF) ===
    st.write("### 🧮 Variance Inflation Factor (VIF)")

    # Ensure all features used in VIF calculation are numeric and valid
    X_vif = sm.add_constant(reg_df[reg_features])
    X_vif = X_vif.select_dtypes(include=[np.number])  # Drop non-numeric columns
    X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()  # Remove inf/nan rows

    # Compute VIFs
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X_vif.columns
    vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    st.dataframe(vif_df)

    st.write("### 🧮 Linear Regression Results")
    # --- Calculate Disparities ---
    merged["faculty_total"] = merged["Grand total"]
    for race in FACULTY_RACE_COLS:
        merged[f"{race}_faculty_pct"] = (merged[FACULTY_RACE_COLS[race]] / merged["faculty_total"]) * 100
        merged[f"{race}_student_pct"] = merged[STUDENT_RACE_COLS[race]]
        merged[f"{race}_disparity"] = merged[f"{race}_faculty_pct"] - merged[f"{race}_student_pct"]

    # --- Regression Model ---

    # Using selected race for disparity and graduation rate

    disparity_column = f"{selected_race}_disparity"
      # Adjust according to your dataset
    
    if selected_race == "Asian":
        grad_rate_column = "Graduation rate, Asian"
    elif selected_race == "Black":
        grad_rate_column = "Graduation rate, Black, non-Hispanic"
    elif selected_race == "Hispanic":
        grad_rate_column = "Graduation rate, Hispanic"
    elif selected_race == "White":
        grad_rate_column = "Graduation rate, White, non-Hispanic"
    elif selected_race == "Two or more":
        grad_rate_column = "Graduation rate, two or more races"
    elif selected_race == "Native American":
        grad_rate_column = "Graduation rate, American Indian or Alaska Native"
    elif selected_race == "Pacific Islander":
        grad_rate_column = "Graduation rate, Native Hawaiian or Other Pacific Islander"

    # Ensure X and y have the same index by aligning them
    X = merged[[disparity_column]].dropna()  # Independent variable (disparity)
    y = merged[grad_rate_column].dropna()  # Dependent variable (graduation rate)

    # Align indices to avoid the mismatch error
    X, y = X.align(y, join='inner', axis=0)

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Show model summary in Streamlit
    st.subheader(f"📈 Regression Model: Disparity vs. Graduation Rate ({selected_race})")
    st.write(model.summary())

    # --- Plotting Regression Line ---
    plt.figure(figsize=(10, 6))
    plt.scatter(X[disparity_column], y, label="Data", color="blue", alpha=0.5)
    plt.plot(X[disparity_column], model.predict(X), label="Fitted Line", color="red", linewidth=2)
    plt.title(f"Disparity vs. Graduation Rate ({selected_race}) - Regression Line")
    plt.xlabel(f"Faculty-Student Disparity ({selected_race})")
    plt.ylabel(f"Graduation Rate ({selected_race})")
    plt.legend()
    st.pyplot(plt)

# --- 📊 Interactive Correlation Matrix --- 
with tabs[4]:
    st.subheader("📈 Interactive Correlation Matrix of Disparities vs. Graduation Rates")
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
    ] + [col for col in GRAD_RATE_COLS if col in merged.columns] + [f"{r}_disparity" for r in FACULTY_RACE_COLS if f"{r}_disparity" in merged.columns]

    # ✅ Safely select columns that exist
    available_features = [col for col in numerical_features + categorical_features if col in merged.columns]
    cluster_df = merged[available_features].dropna()

    cluster_df_encoded = pd.get_dummies(cluster_df, columns=[col for col in categorical_features if col in cluster_df.columns], drop_first=True)
    cluster_df_encoded = cluster_df_encoded.replace([np.inf, -np.inf], np.nan).dropna()

    disparity_columns = [f"{race}_disparity" for race in FACULTY_RACE_COLS if f"{race}_disparity" in cluster_df_encoded.columns]
    grad_rate_columns = [
        "Graduation rate, Black, non-Hispanic", 
        "Graduation rate, White, non-Hispanic", 
        "Graduation rate, two or more races", 
        "Graduation rate, American Indian or Alaska Native",
        "Graduation rate, Native Hawaiian or Other Pacific Islander"
    ]
    grad_rate_columns = [col for col in grad_rate_columns if col in cluster_df_encoded.columns]

    grad_rate_disparity_columns = disparity_columns + grad_rate_columns
    numeric_cols = cluster_df_encoded[grad_rate_disparity_columns]
    numeric_cols = numeric_cols.select_dtypes(include=[np.number])

    corr_matrix = numeric_cols.corr()
    corr_matrix_selected = corr_matrix.loc[grad_rate_columns, disparity_columns]
    rounded_values = np.round(corr_matrix_selected.values, 2)

    x_labels = list(corr_matrix_selected.columns)
    y_labels = list(corr_matrix_selected.index)

    try:
        fig_corr = ff.create_annotated_heatmap(
            z=rounded_values,
            x=x_labels,
            y=y_labels,
            colorscale='YlGnBu',
            showscale=True,
            colorbar_title="Correlation Coefficient"
        )
        fig_corr.update_layout(
            title="Interactive Correlation Matrix of Disparities vs. Graduation Rates",
            xaxis_title="Faculty-Student Disparity",
            yaxis_title="Graduation Rate",
            width=800,
            height=600,
            template="plotly_dark"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error rendering correlation heatmap: {e}")

    st.text("The interactive correlation matrix above shows the relationships between racial disparities in faculty-student composition and graduation rates for each racial group. You can hover over each cell to see the correlation coefficient. Positive correlations indicate that greater disparities in faculty diversity are associated with higher graduation rates, while negative correlations suggest the opposite. This matrix helps understand how faculty-student diversity disparities may influence academic success rates across different racial and ethnic groups.")

    def cramers_v(cat1, cat2):
        confusion_matrix = pd.crosstab(cat1, cat2)
        chi2, p, dof, expected = ss.chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

    def cramers_v_matrix(df, categorical_columns):
        cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 != col2:
                    cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
                else:
                    cramers_v_matrix.loc[col1, col2] = 1.0
        return cramers_v_matrix

    numeric_df = cluster_df_encoded[[col for col in numerical_features if col in cluster_df_encoded.columns]].copy()
    pearson_corr_matrix = numeric_df.corr()

    cramers_v_corr_matrix = cramers_v_matrix(cluster_df, [col for col in categorical_features if col in cluster_df.columns])

    st.write("### 📊 Pearson Correlation Matrix (Numeric Features)")
    st.dataframe(pearson_corr_matrix.style.format("{:.2f}"))

    fig_pearson = ff.create_annotated_heatmap(
        z=pearson_corr_matrix.values.round(2),
        x=list(pearson_corr_matrix.columns),
        y=list(pearson_corr_matrix.index),
        colorscale='PiYG',
        showscale=True,
        colorbar_title="Pearson Correlation"
    )
    fig_pearson.update_layout(
        title="Pearson Correlation Matrix (Numeric Features)",
        xaxis_title="Numeric Features",
        yaxis_title="Numeric Features",
        width=800,
        height=900,
        template="plotly_dark"
    )
    st.plotly_chart(fig_pearson, use_container_width=True)

    st.write("### 📊 Cramér's V Correlation Matrix (Categorical Features)")
    st.dataframe(cramers_v_corr_matrix.style.format("{:.2f}"))

    fig_cramers_v = ff.create_annotated_heatmap(
        z=cramers_v_corr_matrix.values.astype(float).round(2),
        x=list(cramers_v_corr_matrix.columns),
        y=list(cramers_v_corr_matrix.index),
        colorscale='YlGnBu',
        showscale=True,
        colorbar_title="Cramér's V"
    )
    fig_cramers_v.update_layout(
        title="Cramér's V Correlation Matrix (Categorical Features)",
        xaxis_title="Categorical Features",
        yaxis_title="Categorical Features",
        width=800,
        height=600,
        template="plotly_dark"
    )
    st.plotly_chart(fig_cramers_v, use_container_width=True)

    st.text("""
    The Pearson correlation matrix shows the relationships between numeric variables (e.g., faculty-student disparity, graduation rate, etc.).
    The Cramér's V matrix shows the strength of association between categorical features (e.g., institution type, urbanization degree).
    You can hover over the cells to see the exact values for each correlation.
    """)