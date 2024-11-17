
# Script for Hierarchical Clustering
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data
data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.7, random_state=42)

# Apply Hierarchical Clustering
linked = linkage(data, 'ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=30, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=12)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
