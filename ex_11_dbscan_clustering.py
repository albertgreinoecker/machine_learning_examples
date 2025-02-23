import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

def dbscan_clustering():
    # Erstellen eines Beispiel-Datensatzes (Mondförmige Cluster)
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    # DBSCAN Clustering anwenden
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    labels = dbscan.fit_predict(X)

    # Visualisierung der Cluster
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="plasma", s=30, edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("DBSCAN Clustering")
    plt.show()

if __name__ == "__main__":
    dbscan_clustering()
