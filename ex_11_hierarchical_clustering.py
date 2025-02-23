import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# Echten Datensatz laden (Wine Dataset)
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.columns)
# Nur zwei Merkmale für Visualisierung auswählen
df_selected = df[['alcohol', 'malic_acid']]

# Daten normalisieren, weil die Messwerte unterschiedlich skaliert sind
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Dendrogramm erstellen
plt.figure(figsize=(10, 5))
#Zeigt an wie die Cluster zusammengefügt werden (0..cluster1, 1..cluster2, abstand zwischen den Clustern, Anzahl der Punkte im Cluster)
linkage = sch.linkage(df_scaled, method='ward') #Auch möglich: 'single', 'complete', 'average'
plt.title("Dendrogramm für Hierarchisches Clustering")
dendrogram = sch.dendrogram(linkage)
plt.xlabel("Datenpunkte")
plt.ylabel("Abstand (Distanz)")
plt.show()

# Hierarchisches Clustering mit 3 Clustern durchführens
# Ward minimiert die Varianz der Cluster
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = hc.fit_predict(df_scaled)

print(clusters)
# Ergebnisse visualisieren
plt.figure(figsize=(8, 6))
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=clusters, cmap='viridis', edgecolors='k')
plt.xlabel("Alkoholgehalt (normalisiert)")
plt.ylabel("Apfelsäuregehalt (normalisiert)")
plt.title("Hierarchisches Clustering (3 Cluster)")
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=clusters, cmap='viridis', edgecolors='k')
plt.xlabel("Alkoholgehalt (normalisiert)")
plt.ylabel("Apfelsäuregehalt (normalisiert)")
plt.title("Hierarchisches Clustering (3 Cluster)")
plt.show()


plt.xlabel("Alkoholgehalt ")
plt.ylabel("Apfelsäuregehalt")
plt.title("Hierarchisches Clustering (3 Cluster)")
plt.scatter(df_selected[clusters == 0].alcohol, df_selected[clusters == 0].malic_acid, color='red', label='Cluster 0')
plt.scatter(df_selected[clusters == 1].alcohol, df_selected[clusters == 1].malic_acid, color='green', label='Cluster 1')
plt.scatter(df_selected[clusters == 2].alcohol, df_selected[clusters == 2].malic_acid, color='blue', label='Cluster 2')
plt.show()