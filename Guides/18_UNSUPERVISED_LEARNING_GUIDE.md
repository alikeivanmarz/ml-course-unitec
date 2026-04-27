# Unsupervised Learning

Unsupervised learning extracts structure from data without labels. The two dominant tasks are clustering (grouping similar samples) and dimensionality reduction (projecting high-dimensional data into a smaller space that preserves meaningful structure). A third, anomaly detection, identifies samples that deviate from the modelled distribution. This guide covers the algorithms most often used in practice, their evaluation, and the conditions under which each is appropriate.

**Table of Contents**

1. [When to Use Unsupervised Learning](#1-when-to-use-unsupervised-learning)
2. [K-Means and MiniBatchKMeans](#2-k-means-and-minibatchkmeans)
3. [Hierarchical Clustering](#3-hierarchical-clustering)
4. [DBSCAN and HDBSCAN](#4-dbscan-and-hdbscan)
5. [Gaussian Mixture Models](#5-gaussian-mixture-models)
6. [Cluster Evaluation](#6-cluster-evaluation)
7. [PCA — Linear Dimensionality Reduction](#7-pca--linear-dimensionality-reduction)
8. [t-SNE and UMAP — Manifold Learning](#8-t-sne-and-umap--manifold-learning)
9. [Anomaly Detection](#9-anomaly-detection)
10. [Resources](#10-resources)

---

## 1. When to Use Unsupervised Learning

| Goal | Method family |
|------|---------------|
| Group samples by similarity | Clustering (K-Means, hierarchical, DBSCAN, GMM) |
| Reduce dimensionality for storage or downstream models | PCA, autoencoders |
| Visualize high-dimensional data | t-SNE, UMAP, PCA |
| Identify rare or anomalous samples | Isolation Forest, One-Class SVM, LOF |
| Discover topics or themes in text | LDA, NMF |
| Pretrain representations for downstream tasks | Self-supervised learning |

Unsupervised methods produce groupings or projections, not predictions. Their results require interpretation; there is no objective "correct answer" against which to validate them in the absence of labels.

---

## 2. K-Means and MiniBatchKMeans

K-Means partitions samples into `k` clusters by minimizing within-cluster variance. The algorithm alternates between assigning each sample to its nearest centroid and recomputing centroids as cluster means.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=4, random_state=0)

km = KMeans(n_clusters=4, n_init=10, random_state=0)
labels = km.fit_predict(X)
centers = km.cluster_centers_
```

### 2.1 Choosing `k`

The elbow method plots within-cluster sum of squares (`inertia_`) against `k`; the "elbow" point is a heuristic choice.

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
```

The silhouette score (Section 6) is a more rigorous criterion when no obvious elbow exists.

### 2.2 MiniBatchKMeans

For datasets above ~100,000 samples, `MiniBatchKMeans` updates centroids on small random batches and runs an order of magnitude faster.

```python
from sklearn.cluster import MiniBatchKMeans

km = MiniBatchKMeans(n_clusters=10, batch_size=1024, random_state=0).fit(X)
```

### 2.3 Assumptions and Failure Modes

| Assumption | Failure mode when violated |
|------------|----------------------------|
| Clusters are roughly spherical | Stretched or curved clusters split incorrectly |
| Clusters are similarly sized | Small clusters absorbed into large ones |
| Features are scaled comparably | Features with larger ranges dominate the distance |
| `k` is chosen correctly | Either over- or under-segmentation |

Standardize features before applying K-Means.

---

## 3. Hierarchical Clustering

Agglomerative hierarchical clustering builds a tree of nested clusters by repeatedly merging the closest pair. Cutting the tree at a chosen height yields a flat partitioning.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = agg.fit_predict(X)

# Dendrogram for inspection
Z = linkage(X, method="ward")
dendrogram(Z, truncate_mode="lastp", p=20)
plt.show()
```

### 3.1 Linkage Choices

| Linkage | Merge criterion | Tendency |
|---------|-----------------|----------|
| `ward` | Minimize variance within merged cluster | Compact, similarly-sized clusters; default for Euclidean |
| `complete` | Maximum pairwise distance | Compact, well-separated clusters |
| `average` | Average pairwise distance | Balanced trade-off |
| `single` | Minimum pairwise distance | Chaining; sensitive to noise |

Hierarchical clustering scales as $O(n^2)$ in memory and time; impractical for more than ~10,000 samples.

---

## 4. DBSCAN and HDBSCAN

Density-based clustering identifies dense regions separated by lower-density areas. Unlike K-Means, the number of clusters is discovered, and points in low-density regions are labelled as noise.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
# Label -1 indicates noise
```

### 4.1 Parameter Selection

- `eps`: maximum distance between two samples to be considered neighbours.
- `min_samples`: minimum points in a neighbourhood to form a dense region.

A common heuristic for `eps`: plot the distance to the `k`-th nearest neighbour for each point (sorted), and look for the "knee".

### 4.2 HDBSCAN

HDBSCAN extends DBSCAN by varying the density threshold across regions. It eliminates the `eps` parameter and handles clusters of varying density.

```python
# pip install hdbscan
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
labels = clusterer.fit_predict(X)
```

### 4.3 When to Choose Density-Based Methods

| Property | DBSCAN / HDBSCAN | K-Means |
|----------|------------------|---------|
| Cluster shape | Arbitrary | Convex (spherical) |
| Cluster count | Discovered | Specified |
| Noise handling | Built-in (label -1) | Every point assigned |
| Sensitive to scale | Yes — standardize | Yes — standardize |
| Scales to millions | Marginal | Yes (with MiniBatch) |

---

## 5. Gaussian Mixture Models

A Gaussian Mixture Model (GMM) represents the data as a weighted sum of `k` Gaussian distributions. Cluster assignments are probabilistic — each sample has a posterior probability of belonging to each component.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=0)
gmm.fit(X)

labels = gmm.predict(X)            # hard assignment
probs = gmm.predict_proba(X)       # soft assignment
```

### 5.1 Covariance Types

| `covariance_type` | Component shape | Parameters per component |
|-------------------|------------------|--------------------------|
| `full` | Arbitrary ellipsoid | $d(d+1)/2$ |
| `tied` | Single shared covariance | Lower; assumes shared shape |
| `diag` | Axis-aligned ellipsoid | $d$ |
| `spherical` | Sphere | 1 |

`full` is most flexible but has the highest parameter count; restrict to lower-dimensional data or use `diag` / `tied` for higher dimensions.

### 5.2 Selecting `n_components`

Use information criteria to balance fit and complexity:

```python
import numpy as np

ks = range(1, 11)
bic = [GaussianMixture(n_components=k, random_state=0).fit(X).bic(X) for k in ks]
best_k = ks[np.argmin(bic)]
```

BIC penalizes complexity more strongly than AIC; both are valid choices.

---

## 6. Cluster Evaluation

Cluster evaluation splits into two cases: ground-truth labels available, and not.

### 6.1 Without Ground Truth

| Metric | Range | Higher is better | Notes |
|--------|-------|------------------|-------|
| Silhouette | [-1, 1] | Yes | Mean ratio of inter- vs intra-cluster distance |
| Davies–Bouldin | [0, ∞) | No (lower better) | Average similarity of each cluster to its closest |
| Calinski–Harabasz | [0, ∞) | Yes | Ratio of between- to within-cluster dispersion |

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette_score(X, labels)
davies_bouldin_score(X, labels)
calinski_harabasz_score(X, labels)
```

### 6.2 With Ground Truth

| Metric | Range | Notes |
|--------|-------|-------|
| Adjusted Rand Index (ARI) | [-1, 1] | 0 for random, 1 for identical |
| Normalized Mutual Information (NMI) | [0, 1] | Symmetric |
| Homogeneity, Completeness, V-measure | [0, 1] | Decomposable analogues to precision, recall, F1 |

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

adjusted_rand_score(y_true, labels)
normalized_mutual_info_score(y_true, labels)
```

ARI and NMI are invariant to label permutations — cluster index 0 in one set may correspond to index 3 in the other.

---

## 7. PCA — Linear Dimensionality Reduction

Principal Component Analysis projects data onto the directions of maximum variance. The components are orthogonal linear combinations of the original features.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

pca.explained_variance_ratio_     # variance retained per component
pca.explained_variance_ratio_.cumsum()
```

### 7.1 Choosing `n_components`

```python
# Retain enough components to cover 95% of variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
```

### 7.2 PCA Use Cases and Caveats

- **Use cases**: noise reduction, decorrelation, visualization, preprocessing for distance-based methods.
- **Caveats**: assumes linear structure; sensitive to scale (always standardize first); components are not interpretable as original features.

For non-linear structure, use kernel PCA or the manifold-learning methods in Section 8.

---

## 8. t-SNE and UMAP — Manifold Learning

t-SNE and UMAP are non-linear dimensionality reduction methods designed for visualization. Both preserve local neighbourhoods at the cost of distorting global distances.

### 8.1 t-SNE

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_2d = tsne.fit_transform(X)
```

Key parameter: `perplexity` (typical range 5–50) controls the effective number of neighbours considered. t-SNE does not provide a transform for new data — refit is required.

### 8.2 UMAP

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
X_2d = reducer.fit_transform(X)
```

UMAP preserves more global structure than t-SNE, scales better to large datasets, and supports `transform()` on new data.

### 8.3 Comparison

| Property | PCA | t-SNE | UMAP |
|----------|-----|-------|------|
| Linear / non-linear | Linear | Non-linear | Non-linear |
| Preserves global structure | Yes | Poorly | Better than t-SNE |
| Preserves local structure | Limited | Yes | Yes |
| Transforms new data | Yes | No | Yes |
| Scales to >100K samples | Yes | Slow | Yes |
| Use as preprocessing | Yes | No (visualization only) | Sometimes |

---

## 9. Anomaly Detection

Anomaly detection identifies samples that deviate from the modelled distribution. It is appropriate when anomalies are rare, possibly novel, and not well-represented in any training labels.

### 9.1 Isolation Forest

Isolation Forest scores samples by how easily a random tree can isolate them. Anomalies require fewer splits and receive higher scores.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=0)
labels = iso.fit_predict(X)        # -1 for anomalies, 1 for inliers
scores = -iso.score_samples(X)     # higher = more anomalous
```

### 9.2 One-Class SVM

```python
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
labels = ocsvm.fit_predict(X)
```

Suitable for moderate-sized datasets; scales poorly above ~10,000 samples.

### 9.3 Local Outlier Factor

LOF compares the local density of a point to that of its neighbours.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X)
```

LOF is transductive — it does not produce a model for scoring new data unless called with `novelty=True`.

### 9.4 Method Selection

| Method | Best for |
|--------|----------|
| Isolation Forest | Tabular data, moderate-to-large size, mixed feature types |
| One-Class SVM | Small to moderate data with smooth boundary |
| LOF | When local density varies across the feature space |
| Autoencoder reconstruction error | High-dimensional data (images, sequences) |
| Statistical tests (z-score, IQR) | Single-feature, low-dimensional cases |

---

## 10. Resources

- [scikit-learn — Clustering](https://scikit-learn.org/stable/modules/clustering.html) — algorithm comparison, parameters, and metrics.
- [scikit-learn — Decomposition](https://scikit-learn.org/stable/modules/decomposition.html) — PCA, kernel PCA, NMF, and related.
- [scikit-learn — Manifold learning](https://scikit-learn.org/stable/modules/manifold.html) — t-SNE, Isomap, LLE.
- [UMAP documentation](https://umap-learn.readthedocs.io/) — non-linear dimensionality reduction.
- [HDBSCAN documentation](https://hdbscan.readthedocs.io/) — hierarchical density-based clustering.
- [van der Maaten and Hinton, *Visualizing Data using t-SNE* (2008)](https://www.jmlr.org/papers/v9/vandermaaten08a.html) — original t-SNE paper.
- [McInnes et al., *UMAP* (2018)](https://arxiv.org/abs/1802.03426) — original UMAP paper.
- [Liu et al., *Isolation Forest* (2008)](https://ieeexplore.ieee.org/document/4781136) — original Isolation Forest paper.

---

[← Previous: Model Interpretability](17_INTERPRETABILITY_GUIDE.md) | [Index](README.md) | [Next: Time Series and Forecasting →](19_TIME_SERIES_GUIDE.md)
