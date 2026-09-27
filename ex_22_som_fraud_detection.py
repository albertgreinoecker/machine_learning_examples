import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# 1. Laden eines Beispieldatensatzes (fiktiv)
np.random.seed(42)
data = pd.DataFrame(
    {
        "Kreditlimit": np.random.randint(1000, 10000, 500),
        "Saldo": np.random.randint(0, 8000, 500),
        "Transaktionsbetrag": np.random.randint(10, 5000, 500),
        "Anzahl_Transaktionen": np.random.randint(1, 50, 500),
        "Betrug": np.random.choice([0, 1], 500, p=[0.9, 0.1]),  # 10% Betrugsfälle
    }
)

# 2. Datenvorverarbeitung
features = ["Kreditlimit", "Saldo", "Transaktionsbetrag", "Anzahl_Transaktionen"]
X = data[features].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. SOM initialisieren (10x10 Gitter für Clustering)
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=1000)

# 4. Visualisierung der SOM-Ergebnisse
plt.figure(figsize=(8, 8))
for i, x in enumerate(X_scaled):
    winner = som.winner(x)  # Best Matching Unit (BMU)
    marker = "o" if data["Betrug"][i] == 0 else "x"  # 'o' = normal, 'x' = Betrug
    color = "green" if data["Betrug"][i] == 0 else "red"  # Rot für Betrug
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, marker, markerfacecolor=color, markersize=10)

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title("Self-Organizing Map (SOM) für Kreditkartenbetrug")
plt.grid()
plt.show()
