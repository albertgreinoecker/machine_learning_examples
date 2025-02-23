import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Beispiel-Daten generieren (Halbmond-förmige Cluster)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
X = np.append(X, [[2, 2]], axis=0) # Ausreißer hinzufügen

# DBSCAN Clustering durchführen
# eps = max. Entfernung zwischen Punkten, min_samples = min. Clustergröße
dbscan = DBSCAN(eps=0.2, min_samples=5)
# Beinhaltet die Gruppen. Falls keine Gruppe zugeordnet wurde, steht -1
labels = dbscan.fit_predict(X)
print(labels)

# Cluster visualisieren
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
plt.title("DBSCAN Clustering Beispiel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")
plt.show()