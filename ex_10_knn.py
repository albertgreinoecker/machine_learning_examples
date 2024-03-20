import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
# data description: https://archive.ics.uci.edu/dataset/53/iris
# Sepal: Meist bunten Blütenblätter
# Petal: Meist grünen Blätter
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.keys())

# Diese Variablen sind Gegenstand der Vorhersage
print(iris.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Das sind die Labels, die wir bei der Klassfiikation vorhersagen wollen (und bei unsupervised learning ja eigentlich gar nicht wissen müssten)
print(iris.target)
# setosa: Borsten-Schwertlilie
# versicolor: Verschiedenfarbige Schwertlilie
# virginica: Virginische Schwertlilie
print(iris.target_names)


print(iris.target.shape) # 150 Datensaetze
print(iris.data[0])


############################################
# Zeichne


# So bekommt die Colorbar die richtigen Labels
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.scatter(iris.data[:, 0], iris.data[:, 1],
			c=iris.target, cmap=plt.cm.get_cmap('RdYlBu', 3))

plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5) # Zur positionierung der Labels
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);
plt.show()

# Baue das Modell
clf = KMeans(n_clusters=3)

# Die Elemente sind im Moment sortiert nach den Clustern, deshalb zufällig anordnen
zipped = list(zip(iris.data, iris.target))
random.shuffle(zipped)
X,y = zip(*zipped)
clf.fit(X[:100]) #unsupervised, lerne ohne Labels mit den ersten 100 Datensaetzen
print(X)
X = np.array(list(np.vstack(X))) #Mache eine Matrix draus

split = 100
# Vorhersage
predictions = clf.predict(X[split:])	# Vorhersage mit den restlichen 50 Datensaetzen
#probs = clf.predict_proba(clf.predict(X[split:])) # Wahrscheinlichkeiten fuer die Vorhersage
print(list(zip( predictions, y[split:]))) # Vergleiche die Vorhersage mit den tatsaechlichen Labels

plt.scatter(X[split:,0], X[split:, 1],
			c=predictions, cmap=plt.cm.get_cmap('RdYlBu', 3))
plt.show()

print(clf.cluster_centers_) # Die Mittelpunkte der Cluster
# Streuung innerhalb eines Clusters, also dieSumme der quadrierten Abstaende der Datenpunkte zu ihrem naechsten Clusterzentrum
print(clf.inertia_)