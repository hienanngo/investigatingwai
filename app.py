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
st.text("© 2025, CHOE FAN MARQUESES NGO ")
# === 📋 Overview ===
with tabs[0]:
    st.subheader("Introduction")
    st.markdown("""While racial topics are always at the heart of American society’s public discourses, racial equality has proved itself even more significant and prominent on college campuses as it redefines not only college demographic dynamics but also opportunities for many students from less fortunate socio-economic backgrounds.
    From 1961 to 2023, affirmative action has greatly influenced the racial proportions of students in higher education institutions. In 2025, in this post-affirmative-action era, diversity and representation in higher education remain—if not become more—central topics many pay attention to.
    While much focus has been paid to the racial composition of the student body, less scrutiny has been applied to the racial makeup of college faculty. Meanwhile, the diversity of the faculty can, to an extent, dictate the voices in class and in the community.""")

    st.subheader("Research Question")

    st.markdown("> To what extent do racial disparities exist between faculty and students across U.S. colleges, and what patterns can be observed across different institution types?")

    st.subheader("Project Objectives")

    st.markdown("The project aims to analyze this disparity using **dimensionality reduction** and **regression methods**. By looking into the data, the team wishes to uncover systemic trends and highlight areas for more comprehensive **policy intervention** beneficial to all groups of racial minorities.")

    st.subheader("Review of Related Literature")

    st.markdown("""Many previous studies dig into the racial composition of student bodies. However, there is still limited research that focuses on the faculty. Specifically, these studies have found **persistent underrepresentation of racial minorities among college faculty**, even as student representation grows more diverse.For students, racial diversity in U.S. colleges has significantly increased. According to the **National Center for Education Statistics (2020)**:""")
    st.markdown("- The proportion of White students dropped from **77% in 1976** to **55% in 2020**.")
    st.markdown("- **Hispanic students** rose from **6% to 19%**.")
    st.markdown("- **Black students** rose from **10% to 14%** over the same period.")
    st.markdown("Despite this growing diversity, **significant achievement gaps remain**.  The **Education Trust (2018)** highlights that **Black and Latinx students face lower graduation rates**.")

    st.subheader("The Importance of Faculty Diversity")
    st.markdown("""
    Antonio (2001) noted that:
    > While racial diversity improves critical thinking and fosters diverse class spaces, students of color often experience marginalization due to faculty homogeneity.

    Faculty diversity has **not kept pace** with student diversity. According to **NCES (2020)**:
    - Only **6.2%** of full-time faculty were **Black**.
    - **5.5%** were **Latinx**.
    - **2.4%** were **Native American**.
    - **78%** of faculty were **White**.

    This underrepresentation, particularly in **leadership and STEM fields**, affects the academic environment. Researchers argue that **faculty diversity improves student outcomes** by fostering inclusive learning environments (**Freeman et al., 2016**).

    Hurtado’s team (2010) found that:
    > Students perform better in diverse classrooms, suggesting that faculty diversity plays a key role in enhancing the student experience.

    Diverse faculty influence the campus both **academically and socially**.  
    Milem, Chang, and Antonio (2005) noted:
    > When faculty reflect student diversity, students of color are more likely to succeed and feel a sense of belonging.

    In contrast, **Denson and Chang (2009)** pointed out that:
    > A lack of faculty diversity contributes to a negative racial climate.

    **García (2019)** confirms:
    > Especially in elite institutions, students of color report a greater sense of isolation when faculty diversity is lacking.

    ---

    ### Variation Across Institution Types

    Faculty diversity **varies across institutional types**. The **National Academy of Sciences (2018)** reports that:
    - **Public and urban colleges** have more diverse student populations but **lag in faculty diversity**.
    - Elite institutions often lack significant representation of minority faculty in **STEM**, contributing to a **“leaky pipeline”** for underrepresented students (**National Science Foundation, 2018**).

    ---

    ### Conclusion

    Faculty diversity is **crucial** for creating inclusive environments that improve student outcomes, but **significant barriers remain**. Addressing these disparities is essential to fostering a more **equitable and inclusive higher education system**.

    To seek solutions, **our project builds on these studies** by:
    - **Visualizing** disparities across institutions.
    - **Quantitatively examining** how demographic and institutional characteristics relate to the disparity.
    """)

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

    st.markdown("""
    ### Data Source and Description

    Our data are compiled from the **Integrated Postsecondary Education Data System (IPEDS)**, which includes institution-level information on:

    - Faculty and student racial demographics  
    - Institution type  
    - Geographic region  
    - Selectivity  
    - Enrollment size  
    - And more

    This dataset allows us to **compare racial representation between students and faculty at a granular level**.

    Specifically, we calculate a **racial disparity score** by subtracting the **percentage of faculty** from the **percentage of students** for each racial group.

    - The dataset encompasses data for the year **2023** only.  
    - Data was last downloaded on **March 11, 2025**.
                    
    ### Outlining Our Analyses

    1. **Quantify the racial disparity** for each institution and racial group.
    2. Apply **dimensionality reduction techniques** (PCA and UMAP) to explore the structure of the data and identify clustering patterns.
    3. **Color these plots** by racial disparity scores to highlight trends.
    4. Run **linear regressions** to investigate how institutional characteristics (e.g., endowment, region, sector) relate to levels of disparity.
    5. Conduct additional analyses using tools such as:
    - **Interactive Correlation Matrix of Disparities vs. Graduation Rates**
    - **Pearson Correlation Matrix**
    - **Cramér's V Correlation Matrix**

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
    #### Explaining PCA/UMAP Analysis

    Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP) are two dimensionality reduction techniques we employ to visualize the structure of our high-dimensional dataset.

    **PCA** projects the data into a lower-dimensional space by finding directions (principal components) that maximize variance. It helps identify dominant patterns and linear correlations among variables.

    **UMAP**, on the other hand, is a nonlinear technique that preserves local and global structure, making it especially powerful for discovering clusters or manifolds in complex data.

    By reducing the number of dimensions, we can visualize how institutions with similar characteristics cluster and observe where racial disparities may be concentrated.

    #### PCA Analysis:

    PCA reduces complex institutional data into two main components that explain how colleges differ:

    - **PC1** is driven by tuition, financial aid amounts, graduation rate, and faculty salaries. These reflect institutional wealth, selectivity, and resources—factors that often separate elite private schools from public or less-funded institutions.

    - **PC2** is also influenced by financial aid variables, but in a different direction—it distinguishes schools based on how much aid students receive relative to costs and faculty pay, highlighting affordability and support.

    When plotted, we see elite universities cluster in the lower right—they have high tuition, strong graduation rates, and well-paid faculty, but relatively neutral disparity scores for URM, meaning the percent of faculty match that of students. 

    In contrast, large public colleges cluster lower, often with higher negative disparities among Black and Hispanic faculty. This means that in public colleges, there are larger percentages of Black and Hispanic students than percentages of faculty, meaning students at those schools see fewer professors of their same racial background.

    This matters because PCA shows that racial disparities align with structural characteristics, not random chance. By identifying which dimensions drive differences between institutions, PCA helps us see where representation gaps are most concentrated—and why.

    #### PCA Plot Insights: Racial Disparity Patterns

    In the PCA plot, we color each point (institution) by its racial disparity score. This visualization reveals that disparities are not randomly distributed. For instance, institutions with higher disparities for certain races tend to cluster along certain principal components, suggesting shared characteristics such as institutional size, selectivity, or public/private status. For example, some elite private universities appear to exhibit smaller disparities for URM, while many large public institutions show a greater gap—possibly reflecting different hiring practices, tenure structures, or geographic constraints.

    **Black Faculty Disparity**
    High negative disparities cluster among large, less-selective public universities—especially in the lower-left of the PCA plot—highlighting a structural lack of Black faculty representation relative to student populations.

    **Hispanic Faculty Disparity**
    Disparities are concentrated in institutions with high Hispanic student populations but little matching faculty representation. These often appear in the central/top-left PCA region, suggesting insufficient faculty hiring despite strong demand.

    **Asian Faculty Disparity**
    Some elite institutions (upper-right quadrant) show mild overrepresentation of Asian faculty. These are often well-resourced, private, or STEM-focused colleges, where Asian faculty are more concentrated by field.

    **White Faculty Disparity**
    White faculty are consistently overrepresented across most institutions. However, elite universities (upper-right) show less of this overrepresentation, indicating relatively more balanced racial ratios in faculty hiring.

    **Native American Faculty Disparity**
    Disparities are widespread and not clearly clustered, suggesting a universal underrepresentation issue across all institution types.

    **Pacific Islander Faculty Disparity**
    Disparities for Pacific Islander faculty are also widespread but particularly pronounced at institutions with significant Pacific Islander student enrollment—often in western states or Hawaii. These institutions do not show corresponding faculty representation, indicating systemic hiring gaps and underinvestment in faculty pipeline development for this group.

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
    sns.heatmap(corr_matrix, annot=True, cmap="magma", center=0)
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

    # Gracefully handle graduation rate column naming inconsistencies
    grad_rate_column_options = {
        "Asian": ["Graduation rate, Asian", "Graduation rate, Asian/Native Hawaiian/Other Pacific Islander"],
        "Black": ["Graduation rate, Black, non-Hispanic", "Graduation rate, Black"],
        "Hispanic": ["Graduation rate, Hispanic", "Graduation rate, Hispanic or Latino"],
        "White": ["Graduation rate, White, non-Hispanic", "Graduation rate, White"],
        "Two or more": ["Graduation rate, two or more races"],
        "Native American": ["Graduation rate, American Indian or Alaska Native"],
        "Pacific Islander": ["Graduation rate, Native Hawaiian or Other Pacific Islander"]
    }

    grad_rate_column = None
    for col in grad_rate_column_options.get(selected_race, []):
        if col in merged.columns:
            grad_rate_column = col
            break

    if grad_rate_column is None:
        st.error(f"❌ Graduation rate column not found for: '{selected_race}'")
        st.stop()

    # Independent variable
    X = merged[[disparity_column]].dropna()
    y = merged[grad_rate_column].dropna()

    # Align indices
    X, y = X.align(y, join='inner', axis=0)

    # Add intercept
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()

    # Display regression results
    st.subheader(f"📈 Regression Model: Disparity vs. Graduation Rate ({selected_race})")
    st.write(model.summary())

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X[disparity_column], y, label="Data", color="blue", alpha=0.5)
    plt.plot(X[disparity_column], model.predict(X), label="Fitted Line", color="red", linewidth=2)
    plt.title(f"Disparity vs. Graduation Rate ({selected_race}) - Regression Line")
    plt.xlabel(f"Faculty-Student Disparity ({selected_race})")
    plt.ylabel(f"Graduation Rate ({selected_race})")
    plt.legend()
    st.pyplot(plt)

    st.markdown("""
        #### Regression Analysis: Understanding Drivers of Faculty-Student Racial Disparity

        To provide further insight into the drivers of faculty-student racial disparity, we used a regression analysis and found that:

        - **Faculty salary** is consistently associated with lower disparity across several racial groups—suggesting that better-paid institutions tend to have more representative faculties.
        - **Graduation rate** and **endowment per student** also correlate with lower disparities, particularly for **Black** and **Hispanic** faculty.
        - **Enrollment size** and **public status** often predict higher disparity, especially for underrepresented groups.

        These results reinforce that **resources** and **institutional status** are closely tied to faculty-student racial representation.

        These findings suggest that **systemic and structural factors**—not merely pipeline issues—contribute to faculty diversity shortfalls.
    """)

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
    ]

    numerical_features += [col for col in GRAD_RATE_COLS if col in merged.columns]

    if "Graduation rate, Black" in merged.columns and "Graduation rate, Black, non-Hispanic" not in merged.columns:
        numerical_features.append("Graduation rate, Black")
    if "Graduation rate, White" in merged.columns and "Graduation rate, White, non-Hispanic" not in merged.columns:
        numerical_features.append("Graduation rate, White")

    numerical_features += [f"{r}_disparity" for r in FACULTY_RACE_COLS if f"{r}_disparity" in merged.columns]

    all_features = numerical_features + categorical_features
    available_features = [col for col in all_features if col in merged.columns]

    row_thresh = int(0.7 * len(available_features))
    corr_df = merged[available_features].dropna(thresh=row_thresh)

    if corr_df.empty:
        st.error("⚠️ Not enough data after threshold filtering. Please adjust missing data settings.")
        st.stop()

    corr_df_encoded = pd.get_dummies(
        corr_df,
        columns=[col for col in categorical_features if col in corr_df.columns],
        drop_first=True
    )
    corr_df_encoded = corr_df_encoded.replace([np.inf, -np.inf], np.nan).dropna()

    race_order = [
        ("Asian", "Asian"),
        ("Pacific Islander", "Pacific Islander"),
        ("White, non-Hispanic", "White"),
        ("Two or more", "Two or more"),
        ("Black", "Black"),
        ("Black, non-Hispanic", "Black"),
        ("Hispanic", "Hispanic"),
        ("American Indian or Alaska Native", "Native American"),
        ("Native Hawaiian or Other Pacific Islander", "Pacific Islander"),
        ("two or more races", "Two or more")
    ]

    grad_rate_column_options = {
        "Asian": [
            "Graduation rate, Asian",
            "Graduation rate, Asian/Native Hawaiian/Other Pacific Islander"
        ],
        "Black": [
            "Graduation rate, Black",
            "Graduation rate, Black, non-Hispanic"
        ],
        "Hispanic": [
            "Graduation rate, Hispanic",
            "Graduation rate, Hispanic or Latino"
        ],
        "White": [
            "Graduation rate, White",
            "Graduation rate, White, non-Hispanic"
        ],
        "Two or more": [
            "Graduation rate, two or more races",
            "Graduation rate, Two or more"
        ],
        "Native American": [
            "Graduation rate, American Indian or Alaska Native"
        ],
        "Pacific Islander": [
            "Graduation rate, Native Hawaiian or Other Pacific Islander"
        ]
    }

    disparity_columns = []
    grad_rate_columns = []

    for grad_label, disparity_key in race_order:
        grad_col = next((col for col in grad_rate_column_options.get(disparity_key, []) if col in corr_df_encoded.columns), None)
        disparity_col = f"{disparity_key}_disparity"

        if grad_col:
            grad_rate_columns.append(grad_col)

        if disparity_col in corr_df_encoded.columns:
            disparity_columns.append(disparity_col)

    if len(grad_rate_columns) == 0 or len(disparity_columns) == 0:
        st.error("❌ Missing graduation or disparity columns needed for correlation matrix.")
        st.stop()

    grad_rate_disparity_columns = disparity_columns + grad_rate_columns
    numeric_cols = corr_df_encoded[grad_rate_disparity_columns].select_dtypes(include=[np.number])

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
            colorbar_title="Correlation Coefficient",
            annotation_text=[[f"{val:.2f}" for val in row] for row in rounded_values]
        )

        for i, ann in enumerate(fig_corr.layout.annotations):
            row = i // len(x_labels)
            col = i % len(x_labels)
            val = rounded_values[row][col]
            ann.font.color = 'black' if abs(val) < 0.25 else 'white'

        fig_corr.update_layout(
            title="Interactive Correlation Matrix of Disparities vs. Graduation Rates",
            xaxis_title="Faculty-Student Disparity",
            yaxis_title="Graduation Rate",
            width=800,
            height=600,
            template="plotly_dark"
        )

        st.plotly_chart(fig_corr, use_container_width=True, key="heatmap_disparity_grad_2")

    except Exception as e:
        st.error(f"❌ Error rendering correlation heatmap: {e}")



    st.markdown("""
        #### Correlation Matrix Analysis

        To better understand the relationships between key institutional variables and racial disparity scores, we constructed correlation matrices using both the raw and normalized variables in our dataset. These matrices allow us to quickly identify linear associations and highlight which factors may move together across U.S. colleges.

        ##### Graduation Rates vs. Disparities

        - **Strongest positive correlations**:  
        In institutions where Black faculty are more overrepresented relative to Black students (i.e., positive disparity), graduation rates for other racial groups tend to be higher.  
        This might suggest:
            - Institutions making efforts to diversify faculty (especially Black representation) may foster better academic environments across the board.
            - The presence of more diverse faculty could have broader institutional effects beyond just one group.

        - **Mild negative correlations**:  
        As Asian faculty are more overrepresented compared to Asian students, some underrepresented groups’ graduation rates may slightly decline.  
        This could point to:
            - Imbalances in faculty representation that favor one group disproportionately might correlate with lower engagement/support for others.
            - Caution is warranted here—it doesn't imply causation but invites deeper institutional analysis.

        - **Low/Neutral Relationships**:  
        Cross-group effects (e.g., Hispanic disparity influencing White graduation rates) seem minimal compared to within-group or Black faculty influence.  
        Disparities for groups like White, Pacific Islander, and Two or More Races tend to show weaker correlations across graduation rates (values near zero).

        ---

        These relationships are not consistently directional, so they may not carry strong predictive value in isolation.

    """)

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

    cramers_v_corr_matrix = cramers_v_matrix(corr_df, [col for col in categorical_features if col in corr_df.columns])

    numeric_df = corr_df_encoded[[col for col in numerical_features if col in corr_df_encoded.columns]].copy()
    pearson_corr_matrix = numeric_df.corr()

    st.write("### 📊 Pearson Correlation Matrix (Numeric Features)")
    st.dataframe(pearson_corr_matrix.style.format("{:.2f}"))

    z_vals = pearson_corr_matrix.values.round(2)
    font_colors = [['black' if abs(val) < 0.5 else 'white' for val in row] for row in z_vals]

    fig_pearson = ff.create_annotated_heatmap(
        z=pearson_corr_matrix.values.round(2),
        x=list(pearson_corr_matrix.columns),
        y=list(pearson_corr_matrix.index),
        colorscale='PiYG',
        showscale=True,
        colorbar_title="Pearson Correlation",
    )
    for i, ann in enumerate(fig_pearson.layout.annotations):
        val = float(ann.text)
        ann.font.color = 'black' if abs(val) < 0.5 else 'white'
    fig_pearson.update_layout(
        title="Pearson Correlation Matrix (Numeric Features)",
        xaxis_title="Numeric Features",
        yaxis_title="Numeric Features",
        width=800,
        height=900,
        template="plotly_dark"
    )
    st.plotly_chart(fig_pearson, use_container_width=True, key="fig_corr_pearson")

    st.markdown("""
        #### Pearson Correlation Matrix

        This matrix measures how different features correlate. A negative correlation between a group's graduation rate and its disparity score suggests that higher graduation rates do not benefit all groups equally:

        ##### White Students
        - **Correlation with White_dispar**:  
        - Graduation rate (White) vs. White_dispar: **-0.49**
        - **Interpretation**:  
        Even though White students have high graduation rates, the negative correlation with White_dispar suggests gaps may still exist in representation or inclusion among subgroups.

        ##### Asian Students
        - Graduation rate (Asian) vs. Asian_dispar: **-0.22**  
        - Graduation rate (total cohort) vs. Asian_dispar: **-0.18**
        - **Interpretation**:  
        Despite higher average graduation rates, Asian students face moderate disparity, indicating underlying issues in equity or support systems.

        ##### Black Students
        - Black_dispar vs:  
        - Graduation rate (Black): **0.55**  
        - Total enrollment: **0.30**
        - **Interpretation**:  
        There's a positive correlation, meaning as graduation rates rise, disparities actually shrink, which is a promising sign — but the disparity still exists.

        ##### Hispanic Students
        - Hispanic_dispar vs. graduation rates: Correlations are low (**0.02 to 0.10**)
        - **Interpretation**:  
        There's low correlation with graduation rates, suggesting Hispanic students may face challenges unrelated to academic performance — e.g., access, financial aid, or institutional support.

        ##### Native American, Pacific Islander, and Multiracial (Two or More) Students
        - Correlations with their respective disparities are mostly near zero or weakly positive/negative (**-0.05 to 0.15**)
        - **Interpretation**:  
        These groups may not show strong patterns in the data — but this could also be due to underrepresentation or small sample sizes in institutions.

        ---

        #### Big Picture Implications

        High graduation rates don’t equate to equity.  
        - White and Asian groups, while graduating in high numbers, still show significant disparities.  
        - Black students show positive progress, as rising graduation rates seem to help close disparity gaps — a potential model for equity practices.  
        - Systemic barriers may still exist for groups like Hispanic, Native American, and Pacific Islander students — especially in areas beyond graduation statistics.

                """)

    st.write("### 📊 Cramér's V Correlation Matrix (Categorical Features)")
    if not cramers_v_corr_matrix.empty:
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
        st.plotly_chart(fig_cramers_v, use_container_width=True, key="fig_corr_cramers")
    else:
        st.warning("⚠️ Not enough categorical data available to compute Cramér's V correlation matrix.")

    st.markdown("""
        #### What is Cramér’s V?

        Cramér’s V measures the strength of association between two categorical variables, ranging from 0 (no association) to 1 (perfect association). Unlike Pearson's correlation (which is suited for numeric data), this metric is ideal for categorical features like institutional type, locale, etc.

        - **Public/Private vs. Institutional Category** — **Cramér’s V = 0.41**  
        *Moderate association*: Institutional category (e.g., Baccalaureate vs. Research University) and control status (Public vs. Private) are moderately associated — suggesting that certain types of colleges are more likely to be public or private.

        - **Public/Private vs. Urban-centric Locale** — **Cramér’s V = 0.27**  
        *Weak-to-moderate association*: Public/private status has some relation to the institution's geographic setting. Urban campuses may slightly skew private; rural ones may lean public (or vice versa).

        ---

        #### Key Correlations Observed

        ##### A. Racial Disparity Correlations

        - **Positive correlation between disparities of underrepresented minorities (URMs):**  
        Institutions with high Black faculty-student disparity also tend to exhibit low disparities for Hispanic and Native American populations.  
        → This suggests shared structural barriers affecting multiple groups simultaneously.

        - **Negative correlations between White disparity and URM disparities:**  
        As URM representation increases, the share of White faculty may relatively decline.  
        → This highlights a zero-sum dynamic under current hiring practices.

        ---

        #### Interpreting Limitations

        > **Correlation ≠ Causation**

        - Some associations may be driven by latent variables not directly captured in the dataset (e.g., local labor markets, historical hiring practices, state policy environments).
        - Certain variables may be collinear — for example, *enrollment size* and *public/private status*.  
        - To address this, we computed **Variance Inflation Factors (VIF)** to detect and control for **multicollinearity**, ensuring our regression results are more reliable and not distorted by overlapping predictors.

        ---

        #### Correlation Matrix Takeaways

        - The correlation matrices reinforce our key findings:  
        Faculty-student racial disparities are deeply intertwined with **institutional structure**, **region**, and **resources**.
        - These patterns suggest that addressing underrepresentation must occur across **multiple axes simultaneously**, given the strong co-movement of disparities across racial groups.
        - While correlation matrices are **not explanatory on their own**, they provide an essential map for understanding **which institutional factors merit deeper causal investigation**.

                """)