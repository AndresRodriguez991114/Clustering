
# Script for K-Means Clustering
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data
from sklearn.datasets import make_blobs
data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.7, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Visualize results
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)
plt.title("K-Means Clustering Results")
plt.show()
