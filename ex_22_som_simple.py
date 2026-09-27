import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import make_blobs

# 1. Generiere zufällige Daten
X, y = make_blobs(n_samples=500, centers=3)
print(X)
print(y)
# 2. Initialisiere das SOM-Netzwerk
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)

# 3. Trainiere das SOM
som.train_random(X, num_iteration=1000)

# 4. Visualisierung der SOM-Gewichte
plt.figure(figsize=(8, 8))
for i, (x, y_label) in enumerate(zip(X, y)):
    winner = som.winner(x)  # Best Matching Unit (BMU)
    plt.text(
        winner[0] + 0.5,
        winner[1] + 0.5,
        str(y_label),
        color=plt.cm.rainbow(y_label / 2),
        fontdict={"weight": "bold", "size": 12},
    )

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title("Self-Organizing Map (SOM) Clustering")
plt.grid()
plt.show()
