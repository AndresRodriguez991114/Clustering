
# Script for DBSCAN Clustering
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate synthetic data
from sklearn.datasets import make_moons
data, _ = make_moons(n_samples=100, noise=0.05, random_state=42)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(data)

# Visualize results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
plt.title("DBSCAN Clustering Results")
plt.show()
