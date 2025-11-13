from tensorflow.keras.datasets import reuters

# Nur die 10.000 häufigsten Wörter verwenden
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

print(len(x_train), "Trainingsbeispiele")
print(len(x_test), "Testbeispiele")
print(max(y_train), "Kategorien")

label_names = [
    "cocoa", "grain", "veg-oil", "earn", "acq", "wheat", "corn", "crude",
    "money-fx", "interest", "ship", "trade", "reserves", "cotton", "coffee",
    "sugar", "gold", "tin", "strategic-metal", "livestock", "retail", "ipi",
    "iron-steel", "rubber", "heat", "jobs", "lei", "bop", "carcass",
    "money-supply", "alum", "oilseed", "meal-feed", "cpi", "housing",
    "rubber", "zinc", "nickel", "orange", "pet-chem", "dlr", "gas", "silver",
    "wpi", "strategic-reserves", "wheat-germ"
]

print(f"Beispielklasse: {y_train[0]} {label_names[y_train[0]]}")