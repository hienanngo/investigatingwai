import pandas as pd
import streamlit as st

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
with st.spinner("📥 Loading data..."):
    staff_df = load_data_from_gsheet(STAFF_SHEET_ID)
    student_df = load_data_from_gsheet(STUDENT_SHEET_ID)

# --- Filter to Instructional Staff ---
staff_df = staff_df[
    staff_df["Occupation and full- and part-time status"] == "Instructional staff"
].copy()

# --- Merge on unitid and year ---
merged = pd.merge(student_df, staff_df, on=["unitid", "year"], how="inner")

# --- Clean Filters ---
valid_title_iv = "Title IV postsecondary institution"
valid_categories = [
    "Degree-granting, primarily baccalaureate or above",
    "Degree-granting, associate's and certificates",
    "Degree-granting, not primarily baccalaureate or above",
    "Degree-granting, graduate with no undergraduate degrees"
]

# Remove bad or missing values
cleaned = merged[
    (merged["Postsecondary and Title IV institution indicator"] == valid_title_iv) &
    (merged["Institutional category"].isin(valid_categories)) &
    (~merged["Degree of urbanization (Urban-centric locale)"].isin(["{Not available}"])) &
    (~merged["Institution size category"].isin(["Not reported", "Not applicable"]))
].copy()

# --- Map Public/Private with Private not-for-profit as its own category ---
control_map = {
    "Public": "Public",
    "Private for-profit": "Private for-profit",
    "Private not-for-profit": "Private not-for-profit"
}
cleaned["Public/Private"] = cleaned["Control of institution"].map(control_map)

# --- Race Mappings ---
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
cleaned["faculty_total"] = cleaned["Grand total"]

for race in faculty_race_cols:
    cleaned[f"{race}_faculty_pct"] = (cleaned[faculty_race_cols[race]] / cleaned["faculty_total"]) * 100
    cleaned[f"{race}_student_pct"] = cleaned[student_race_cols[race]]
    cleaned[f"{race}_disparity"] = cleaned[f"{race}_faculty_pct"] - cleaned[f"{race}_student_pct"]

# --- Graduation Rates by Race Columns ---
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

# --- Streamlit UI ---
st.subheader("🧮 Explore Racial Disparities & Institution Features")
selected_race = st.selectbox("Choose a racial group", list(faculty_race_cols.keys()))

display_cols = [
    "institution name_x",
    f"{selected_race}_faculty_pct",
    f"{selected_race}_student_pct",
    f"{selected_race}_disparity",
    "State abbreviation",
    "Public/Private",
    "Institutional category",
    "Degree of urbanization (Urban-centric locale)",
    "Total  enrollment",
    "Tuition and fees, 2023-24",
    "Percent admitted - total",
    "Graduation rate, total cohort"
] + grad_rate_cols

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Institutions")

# Filter: State
state_options = ["All"] + sorted(cleaned["State abbreviation"].dropna().unique())
selected_state = st.sidebar.selectbox("State", state_options)

# Filter: Control Type
control_options = ["All"] + sorted(cleaned["Public/Private"].dropna().unique())
selected_control = st.sidebar.selectbox("Institution Control", control_options)

# Filter: Degree Type
degree_options = ["All"] + sorted(cleaned["Institutional category"].dropna().unique())
selected_degree = st.sidebar.selectbox("Degree Type", degree_options)

# Filter: Urbanicity
urban_options = ["All"] + sorted(cleaned["Degree of urbanization (Urban-centric locale)"].dropna().unique())
selected_urban = st.sidebar.selectbox("Urbanicity", urban_options)

# --- Apply filters to dataframe ---
filtered_df = cleaned.copy()
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["State abbreviation"] == selected_state]

if selected_control != "All":
    filtered_df = filtered_df[filtered_df["Public/Private"] == selected_control]

if selected_degree != "All":
    filtered_df = filtered_df[filtered_df["Institutional category"] == selected_degree]

if selected_urban != "All":
    filtered_df = filtered_df[filtered_df["Degree of urbanization (Urban-centric locale)"] == selected_urban]


# --- Slider: Total Enrollment ---
min_enroll = int(cleaned["Total  enrollment"].min())
max_enroll = int(cleaned["Total  enrollment"].max())
enrollment_range = st.sidebar.slider(
    "Total Enrollment Range",
    min_value=min_enroll,
    max_value=max_enroll,
    value=(min_enroll, max_enroll),
    step=100,
    key="enrollment_slider"
)

# --- Slider: Tuition ---
min_tuition = int(cleaned["Tuition and fees, 2023-24"].min())
max_tuition = int(cleaned["Tuition and fees, 2023-24"].max())
tuition_range = st.sidebar.slider(
    "Tuition Range ($)",
    min_value=min_tuition,
    max_value=max_tuition,
    value=(min_tuition, max_tuition),
    step=500,
    key="tuition_slider"
)

filtered_df = filtered_df[
    (filtered_df["Total  enrollment"] >= enrollment_range[0]) &
    (filtered_df["Total  enrollment"] <= enrollment_range[1])
]

filtered_df = filtered_df[
    (filtered_df["Tuition and fees, 2023-24"] >= tuition_range[0]) &
    (filtered_df["Tuition and fees, 2023-24"] <= tuition_range[1])
]

display_df = filtered_df[display_cols].rename(columns={
    "institution name_x": "Institution",
    "Public/Private": "Institution Control",
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


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.header("🧠 Principal Component Analysis (PCA)")

# --- Step 1: Prepare PCA Features ---

# Numerical columns
numerical_features = [
    "Total  enrollment",
    "Tuition and fees, 2023-24",
    "Percent admitted - total",
    "Graduation rate, total cohort"
] + grad_rate_cols + [f"{race}_disparity" for race in faculty_race_cols]

# Categorical columns
categorical_features = [
    "Public/Private",
    "Degree of urbanization (Urban-centric locale)",
    "Institutional category"
]

# Select and copy subset
pca_data = filtered_df[numerical_features + categorical_features].copy()

# --- Step 2: Drop columns with too many NaNs (over 20% missing) ---
column_thresh = 0.8 * len(pca_data)
pca_data = pca_data.dropna(axis=1, thresh=column_thresh)

# --- Step 3: Drop rows with remaining NaNs ---
pca_data = pca_data.dropna()

# --- Step 4: Encode categorical columns ---
cat_cols = [col for col in categorical_features if col in pca_data.columns]
pca_data = pd.get_dummies(pca_data, columns=cat_cols, drop_first=True)

# --- Step 5: Standardize data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_data)

# ✅ This is the function (defined above)
@st.cache_data(show_spinner="🔍 Computing PCA, UMAP, and Clustering...")
def compute_embeddings(X_scaled, n_clusters):
    from sklearn.decomposition import PCA
    import umap.umap_ as umap
    from sklearn.cluster import KMeans

    # PCA
    pca_result = PCA(n_components=2).fit_transform(X_scaled)

    # UMAP
    umap_result = umap.UMAP(random_state=42).fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    return pca_result, umap_result, cluster_labels

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Clustering Controls")

# Cluster count slider
num_clusters = st.sidebar.slider(
    "Number of clusters", min_value=2, max_value=10, value=5, step=1, key="num_clusters"
)


# --- CLEAN & PREP DATA ---
selected_cols = categorical_features + selected_features + ["institution name_x"]
thresh = 0.8 * len(filtered_df)
cluster_df = filtered_df[selected_cols].dropna(axis=1, thresh=thresh)
cluster_df = cluster_df.dropna()

# Encode categorical variables
cluster_df_encoded = pd.get_dummies(cluster_df, columns=[col for col in categorical_features if col in cluster_df.columns], drop_first=True)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df_encoded.drop(columns=["institution name_x"]))

# --- Compute everything (cached) ---
pca_result, umap_result, cluster_labels = compute_embeddings(X_scaled, num_clusters)

# Add results to your DataFrame
cluster_df_encoded["PCA_1"] = pca_result[:, 0]
cluster_df_encoded["PCA_2"] = pca_result[:, 1]
cluster_df_encoded["UMAP_1"] = umap_result[:, 0]
cluster_df_encoded["UMAP_2"] = umap_result[:, 1]
cluster_df_encoded["Cluster"] = cluster_labels

# --- Step 7: Explained Variance Table + Plot ---
st.subheader("🔍 Variance Explained by Components")

explained_var = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "Variance Explained": pca.explained_variance_ratio_
})

st.dataframe(explained_var.head(10))

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(
    x=range(1, len(pca.explained_variance_ratio_)+1),
    y=pca.explained_variance_ratio_.cumsum(),
    marker="o"
)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("📈 PCA: Cumulative Variance Explained")
ax.grid(True)
st.pyplot(fig)

# --- Step 8: Top contributing features ---
components_df = pd.DataFrame(pca.components_, columns=pca_data.columns)

st.subheader("📌 Top Features in PC1 (Most variance):")
st.dataframe(components_df.iloc[0].sort_values(ascending=False).head(10))

st.subheader("📌 Top Features in PC2:")
st.dataframe(components_df.iloc[1].sort_values(ascending=False).head(10))


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.header("📈 Correlation with Faculty Representation Disparity")

# --- Select a disparity variable to analyze ---
selected_corr_race = st.selectbox(
    "Choose a racial disparity to explore",
    list(faculty_race_cols.keys()),
    key="correlation_selector"
)

disparity_var = f"{selected_corr_race}_disparity"

# --- Pick relevant numeric institutional features ---
corr_features = [
    "Tuition and fees, 2023-24",
    "Total  enrollment",
    "Percent admitted - total",
    "Graduation rate, total cohort",
]

# One-hot encode Public/Private
public_private_dummies = pd.get_dummies(filtered_df["Public/Private"], prefix="Control", drop_first=True)
corr_df = pd.concat([filtered_df[[disparity_var] + corr_features], public_private_dummies], axis=1)


# --- Make sure the data is clean ---
corr_df = filtered_df[[disparity_var] + corr_features].copy()
corr_df = corr_df.dropna()

# --- Compute correlation ---
correlations = corr_df.corr()[[disparity_var]].drop(disparity_var).sort_values(by=disparity_var, ascending=False)

st.subheader(f"📊 Correlation with {selected_corr_race} Disparity")
st.dataframe(correlations.style.format("{:.2f}"))

# --- Plot heatmap ---
fig, ax = plt.subplots()
sns.heatmap(correlations.T, annot=True, cmap="coolwarm", center=0)
st.pyplot(fig)

# --- Optional: Run simple linear regression ---
with st.expander("🧮 Show linear regression summary"):
    X = corr_df.drop(columns=[disparity_var])
    y = corr_df[disparity_var]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.text(model.summary())

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
import plotly.express as px

st.header("🧬 Dimensionality Reduction & Clustering")

# Feature group selector
feature_groups = st.sidebar.multiselect(
    "Include in clustering:",
    ["Institutional features", "Graduation rates", "Racial disparities"],
    default=["Institutional features", "Graduation rates", "Racial disparities"]
)

# --- FEATURE DEFINITIONS ---
institutional_features = [
    "Total  enrollment",
    "Tuition and fees, 2023-24",
    "Percent admitted - total"
]

grad_features = [
    "Graduation rate, total cohort",
    "Graduation rate, American Indian or Alaska Native",
    "Graduation rate, Asian/Native Hawaiian/Other Pacific Islander",
    "Graduation rate, Asian",
    "Graduation rate, Native Hawaiian or Other Pacific Islander",
    "Graduation rate, Black, non-Hispanic",
    "Graduation rate, Hispanic",
    "Graduation rate, White, non-Hispanic",
    "Graduation rate, two or more races"
]

disparity_features = [f"{race}_disparity" for race in faculty_race_cols]

categorical_cols = [
    "Public/Private",
    "Degree of urbanization (Urban-centric locale)",
    "Institutional category"
]

# --- BUILD SELECTED FEATURE SET ---
selected_features = []
if "Institutional features" in feature_groups:
    selected_features += institutional_features
if "Graduation rates" in feature_groups:
    selected_features += grad_features
if "Racial disparities" in feature_groups:
    selected_features += disparity_features


# --- PCA ---
pca_result = PCA(n_components=2).fit_transform(X_scaled)
cluster_df_encoded["PCA_1"] = pca_result[:, 0]
cluster_df_encoded["PCA_2"] = pca_result[:, 1]

# --- UMAP ---
umap_result = umap.UMAP(random_state=42).fit_transform(X_scaled)
cluster_df_encoded["UMAP_1"] = umap_result[:, 0]
cluster_df_encoded["UMAP_2"] = umap_result[:, 1]

# --- KMEANS CLUSTERING ---
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_df_encoded["Cluster"] = kmeans.fit_predict(X_scaled)
cluster_df_encoded["Institution"] = cluster_df["institution name_x"]

# Bring original disparity columns back for plotting
for race, col in faculty_race_cols.items():
    disparity_col = f"{race}_disparity"
    if disparity_col in cluster_df.columns:
        cluster_df_encoded[disparity_col] = cluster_df[disparity_col].values


st.header("🎯 Explore Clusters by Racial Disparity")


disparity_races = {
    "Asian": "Asian_disparity",
    "Black": "Black_disparity",
    "Hispanic": "Hispanic_disparity",
    "White": "White_disparity",
    "Two or more": "Two or more_disparity",
    "Native American": "Native American_disparity",
    "Pacific Islander": "Pacific Islander_disparity"
}

# Get all disparity values across races to estimate scale
all_disparities = cluster_df_encoded[[f"{r}_disparity" for r in faculty_race_cols]].stack()

# Clip to 1st–99th percentiles to avoid extreme outliers
vmin = np.percentile(all_disparities, 1)
vmax = np.percentile(all_disparities, 99)
abs_max = max(abs(vmin), abs(vmax))  # symmetric range

# Calculate a consistent color range based on 1st–99th percentiles
all_disparities = cluster_df_encoded[[f"{r}_disparity" for r in faculty_race_cols if f"{r}_disparity" in cluster_df_encoded.columns]].stack()
vmin = np.percentile(all_disparities, 1)
vmax = np.percentile(all_disparities, 99)
abs_max = max(abs(vmin), abs(vmax))  # Symmetric color scale

# Create tabs
tabs = st.tabs(list(disparity_races.keys()))

for i, (race_label, race_column) in enumerate(disparity_races.items()):
    with tabs[i]:
        st.subheader(f"{race_label} Disparity by Cluster")

        try:
            plot_df = cluster_df_encoded[[
                race_column, "PCA_1", "PCA_2", "UMAP_1", "UMAP_2", "Institution", "Cluster"
            ]].copy()

            # Clean
            plot_df[race_column] = pd.to_numeric(plot_df[race_column], errors="coerce")
            plot_df = plot_df.dropna(subset=[race_column])

            if plot_df.empty:
                st.warning(f"⚠️ No data to plot for {race_label}.")
                continue

            # Debug output
            st.caption(f"Data range: {plot_df[race_column].min():.1f} to {plot_df[race_column].max():.1f} — {plot_df.shape[0]} rows")

            # PCA Plot
            fig_pca = px.scatter(
                plot_df,
                x="PCA_1", y="PCA_2",
                color=race_column,
                color_continuous_scale="RdBu",
                range_color=[-abs_max, abs_max],
                hover_data=["Institution", "Cluster"],
                title=f"PCA View – {race_label} Disparity"
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            # UMAP Plot
            fig_umap = px.scatter(
                plot_df,
                x="UMAP_1", y="UMAP_2",
                color=race_column,
                color_continuous_scale="RdBu",
                range_color=[-abs_max, abs_max],
                hover_data=["Institution", "Cluster"],
                title=f"UMAP View – {race_label} Disparity"
            )
            st.plotly_chart(fig_umap, use_container_width=True)

            # Cluster mean table
            avg_disp = cluster_df_encoded.groupby("Cluster")[race_column].mean().reset_index()
            avg_disp.columns = ["Cluster", f"Avg {race_label} Disparity"]
            st.dataframe(avg_disp)

        except Exception as e:
            st.error(f"❌ Error rendering {race_label} tab: {e}")




# --- CLUSTER SIZE SUMMARY ---
cluster_counts = cluster_df_encoded["Cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["Cluster", "Number of Universities"]
st.subheader("📊 Number of Universities per Cluster")
st.dataframe(cluster_counts)

all_disparities = cluster_df_encoded[[f"{r}_disparity" for r in faculty_race_cols]].stack()
st.write(f"Disparity range: {all_disparities.min():.1f} to {all_disparities.max():.1f}")
