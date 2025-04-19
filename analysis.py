# analysis.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
import statsmodels.api as sm
import pandas as pd

def prepare_cluster_data(df, features):
    df_clean = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    return X_scaled, df_clean.index

def run_dimensionality_reduction(X_scaled, n_clusters=4):
    pca_model = PCA(n_components=2)
    pca_result = pca_model.fit_transform(X_scaled)

    umap_result = umap.UMAP(random_state=42).fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    return pca_model, pca_result, umap_result, cluster_labels

def run_logistic_regressions(df, target_var, features):
    df = df[[target_var] + features].dropna()
    df["binary_target"] = (df[target_var] > 0).astype(int)

    X = df[features]
    X = sm.add_constant(X)
    y = df["binary_target"]

    model = sm.Logit(y, X).fit(disp=False)
    summary_df = pd.DataFrame({
        "Feature": model.params.index,
        "Coefficient": model.params.values,
        "p-value": model.pvalues
    })
    return summary_df
