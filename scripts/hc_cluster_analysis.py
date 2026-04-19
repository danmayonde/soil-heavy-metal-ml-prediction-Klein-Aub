import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')


# ========================================================================
# Data Loading and Data Preprocessing
# ========================================================================

df = ((pd.read_csv("../Data/BData.csv")).drop(columns=["Sample_ID"])).dropna()

# Transform Aspect to Sine and Cosine
aspect_rad = np.radians(df["Aspect (˚)"])   # Convert to radians

# Insert the two new features next to "Aspect (˚)"
df.insert(5, "aspect_sin", np.sin(aspect_rad))
df.insert(6, "aspect_cos", np.cos(aspect_rad))

# Drop "Aspect (˚)" from the data frame
df.drop(["Aspect (˚)"], axis=1, inplace=True)

# Encode the Categorical Feature "Land Use" to 1 for "residential" and 0 for "barren"
df["Land Use"] = (df["Land Use"] == "residential").astype(int)


# Log10 Transformation
print('/.. run log transformation...', '\n')
# Variables to transform
feat_transform = ["Clay (%)", "TDS (mg/L)", "SOM (g/kg)", 'NDVI']

# log10 transformation of highly skewed input variables
for feature in feat_transform:
    df[feature] = np.log10(df[feature])

print("=" * 40)
print("Log10 Transformed DataFrame")
print("=" * 40)
print(f'\n {df[feat_transform].head(10)}\n')

print('feature transformation complete...') 

targets = ['Cu (mg/kg)', 'Zn (mg/kg)', 'Pb (mg/kg)']
redundant_features = ["Longitude", "Latitude", "Dist_Main_Road  (m)", "EC (µs/cm)", "Sand (%)"]

features_to_drop = targets + redundant_features

# Create X Data and y

X = df.drop(columns=features_to_drop).copy()

y = df[targets]

feature_cols = X.columns

y_Cu = df["Cu (mg/kg)"]
y_Zn = df["Zn (mg/kg)"]
y_Pb = df["Pb (mg/kg)"]

# ========================================================================
# Data Splitting into Train and Test Sets
# ========================================================================

# Split the DataSet into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sperating targets array into different arrays
y_train_Cu = y_train[targets[0]]
y_train_Zn = y_train[targets[1]]
y_train_Pb = y_train[targets[2]]

y_test_Cu = y_test[targets[0]]
y_test_Zn = y_test[targets[1]]
y_test_Pb = y_test[targets[2]]

scaler = StandardScaler()
X_train_num = X_train.drop("Land Use", axis=1).copy()
X_train_scaled = scaler.fit_transform(X_train_num)

land_use_arr = np.asarray(X_train["Land Use"]).reshape(-1, 1)

X_train_scaled_final = np.append(X_train_scaled, land_use_arr, axis=1)



# ============================================================
# Create dendrogram
# ============================================================

# Compute linkage matrix (for dendrogram)
linkage_matrix = linkage(X_train_scaled_final, method='ward')

# Create dendrogram plot
plt.figure(figsize=(16, 10))
dendrogram(
    linkage_matrix,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Distance (Ward)', fontsize=12)
plt.axhline(y=10, color='r', linestyle='--', label='Potential cut (k=6)')
plt.axhline(y=15, color='g', linestyle='--', label='Potential cut (k=2)')
plt.legend()
plt.tight_layout()
plt.savefig(f'../clusters/dendrogram_cluster_selection.png', dpi=300)
plt.close()

# ============================================================
# Calculate metrics for candidate k values
# ============================================================

# Test different numbers of clusters
k_range = range(2, 8) 
results = []

for k in k_range:
    # Fit clustering
    clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = clusterer.fit_predict(X_train_scaled_final)
    
    # Calculate metrics
    silhouette = silhouette_score(X_train_scaled_final, labels)
    davies_bouldin = davies_bouldin_score(X_train_scaled_final, labels)
    calinski_harabasz = calinski_harabasz_score(X_train_scaled_final, labels)
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    min_cluster_size = counts.min()
    max_cluster_size = counts.max()
    size_ratio = max_cluster_size / min_cluster_size
    
    results.append({
        'k': k,
        'Silhouette': silhouette,
        'Davies-Bouldin': davies_bouldin,
        'Calinski-Harabasz': calinski_harabasz,
        'Min Size': min_cluster_size,
        'Max Size': max_cluster_size,
        'Size Ratio': size_ratio
    })

# Display results
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("QUANTITATIVE METRICS FOR DIFFERENT k VALUES")
print("="*70)
print(results_df.to_string(index=False))

# ============================================================
# Identify best k
# ============================================================

# Find best by each metric
best_silhouette = results_df.loc[results_df['Silhouette'].idxmax()]
best_db = results_df.loc[results_df['Davies-Bouldin'].idxmin()]
best_ch = results_df.loc[results_df['Calinski-Harabasz'].idxmax()]

print("\n" + "="*70)
print("BEST k BY EACH METRIC")
print("="*70)
print(f"Best Silhouette Score: k={best_silhouette['k']:.0f} (score={best_silhouette['Silhouette']:.3f})")
print(f"Best Davies-Bouldin: k={best_db['k']:.0f} (score={best_db['Davies-Bouldin']:.3f})")
print(f"Best Calinski-Harabasz: k={best_ch['k']:.0f} (score={best_ch['Calinski-Harabasz']:.1f})")

# ============================================================
# Visualization - Elbow plots
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Silhouette
axes[0].plot(results_df['k'], results_df['Silhouette'], 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0].set_ylabel('Silhouette Score', fontsize=11)
axes[0].set_title('Silhouette Score', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=best_silhouette['k'], color='r', linestyle='--', alpha=0.7, label=f'Best k={best_silhouette["k"]:.0f}')
axes[0].legend()

# Davies-Bouldin
axes[1].plot(results_df['k'], results_df['Davies-Bouldin'], 'o-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[1].set_title('Davies-Bouldin Index', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=best_db['k'], color='r', linestyle='--', alpha=0.7, label=f'Best k={best_db["k"]:.0f}')
axes[1].legend()

# Calinski-Harabasz
axes[2].plot(results_df['k'], results_df['Calinski-Harabasz'], 'o-', linewidth=2, markersize=8, color='green')
axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[2].set_ylabel('Calinski-Harabasz Index', fontsize=11)
axes[2].set_title('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=best_ch['k'], color='r', linestyle='--', alpha=0.7, label=f'Best k={best_ch["k"]:.0f}')
axes[2].legend()

plt.tight_layout()
plt.savefig(f'../clusters/cluster_metrics_comparison.png', dpi=300)
plt.close()

# ============================================================
# Final recommendation
# ============================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Check consensus
votes = [best_silhouette['k'], best_db['k'], best_ch['k']]
from collections import Counter
vote_counts = Counter(votes)
most_common_k = vote_counts.most_common(1)[0][0]
consensus_count = vote_counts[most_common_k]

if consensus_count >= 2:
    print(f"✓ STRONG CONSENSUS: k={most_common_k:.0f}")
    print(f"  ({consensus_count}/3 metrics agree)")
    recommended_k = most_common_k
else:
    print(f"⚠️ NO CONSENSUS: Metrics disagree")
    print(f"  Silhouette suggests: k={best_silhouette['k']:.0f}")
    print(f"  Davies-Bouldin suggests: k={best_db['k']:.0f}")
    print(f"  Calinski-Harabasz suggests: k={best_ch['k']:.0f}")
    print(f"  Consider: k=2 (simple) or k=6 (more granular)")
    recommended_k = int(best_silhouette['k'])  # Default to silhouette

# Check cluster size balance
recommended_row = results_df[results_df['k'] == recommended_k].iloc[0]
if recommended_row['Size Ratio'] > 3.0:
    print(f"\n⚠️ WARNING: k={recommended_k:.0f} creates imbalanced clusters")
    print(f"  Size ratio: {recommended_row['Size Ratio']:.2f}")
    print(f"  Largest cluster: {recommended_row['Max Size']:.0f} samples")
    print(f"  Smallest cluster: {recommended_row['Min Size']:.0f} samples")
    print(f"  Consider: k=2 for better balance")

# Final recommendation
print(f"\n{'='*70}")
print(f"FINAL RECOMMENDATION: k={recommended_k:.0f}")
print(f"{'='*70}")
print(f"Metrics for k={recommended_k:.0f}:")
print(f"  Silhouette: {recommended_row['Silhouette']:.3f}")
print(f"  Davies-Bouldin: {recommended_row['Davies-Bouldin']:.3f}")
print(f"  Calinski-Harabasz: {recommended_row['Calinski-Harabasz']:.1f}")
print(f"  Cluster sizes: {recommended_row['Min Size']:.0f} to {recommended_row['Max Size']:.0f}")
print(f"\nCombined with dendrogram visual inspection!")

# ============================================================
# Plot Clusters
# ============================================================
hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')

y_hc = hc.fit_predict(X_train_scaled_final)

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled_final)

plt.figure(figsize=(8,6))

n_clusters = len(np.unique(y_hc))

for cluster in range(n_clusters):
    
    plt.scatter(
        X_pca[y_hc == cluster, 0],
        X_pca[y_hc == cluster, 1],
        label=f"Cluster {cluster+1}",
        s=60
    )

centroids = np.array([
    X_train_scaled_final[y_hc == i].mean(axis=0)
    for i in range(n_clusters)
])

centroids_pca = pca.transform(centroids)

plt.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    marker="X",
    s=250,
    c="black",
    label="Centroids"
)

plt.title("Graphical Visulisation of Clusters", fontweight='bold', fontstyle='italic')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("../clusters/scatter_plot_clusters.png")
plt.close()