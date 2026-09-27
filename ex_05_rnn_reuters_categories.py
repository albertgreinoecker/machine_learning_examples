from tensorflow.keras.datasets import reuters

# Nur die 10.000 häufigsten Wörter verwenden
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

print(len(x_train), "Trainingsbeispiele")
print(len(x_test), "Testbeispiele")
print(max(y_train), "Kategorien")

label_names = [
    "cocoa",
    "grain",
    "veg-oil",
    "earn",
    "acq",
    "wheat",
    "corn",
    "crude",
    "money-fx",
    "interest",
    "ship",
    "trade",
    "reserves",
    "cotton",
    "coffee",
    "sugar",
    "gold",
    "tin",
    "strategic-metal",
    "livestock",
    "retail",
    "ipi",
    "iron-steel",
    "rubber",
    "heat",
    "jobs",
    "lei",
    "bop",
    "carcass",
    "money-supply",
    "alum",
    "oilseed",
    "meal-feed",
    "cpi",
    "housing",
    "rubber",
    "zinc",
    "nickel",
    "orange",
    "pet-chem",
    "dlr",
    "gas",
    "silver",
    "wpi",
    "strategic-reserves",
    "wheat-germ",
]

label_names_de = [
    "Kakao",
    "Getreide",
    "Pflanzenöl",
    "Gewinn",
    "Übernahme",
    "Weizen",
    "Mais",
    "Rohöl",
    "Devisen",
    "Zinsen",
    "Schifffahrt",
    "Handel",
    "Währungsreserven",
    "Baumwolle",
    "Kaffee",
    "Zucker",
    "Gold",
    "Zinn",
    "strategisches Metall",
    "Viehbestand",
    "Einzelhandel",
    "Industrieproduktion",
    "Eisen und Stahl",
    "Kautschuk",
    "Heizung",
    "Arbeitsplätze",
    "Frühindikator (LEI)",
    "Leistungsbilanz",
    "Schlachtkörper",
    "Geldmenge",
    "Aluminium",
    "Ölsaaten",
    "Futtermittel",
    "Verbraucherpreisindex (VPI)",
    "Wohnungsbau",
    "Kautschuk",
    "Zink",
    "Nickel",
    "Orange",
    "Petrochemie",
    "US-Dollar",
    "Gas",
    "Silber",
    "Großhandelspreisindex (GPI)",
    "strategische Reserven",
    "Weizenkeim",
]

print(f"Beispielklasse: {y_train[0]} {label_names_de[y_train[0]]}")


import kagglehub

# Download latest version
path = kagglehub.dataset_download("sunilthite/text-document-classification-dataset")

print("Path to dataset files:", path)


path = kagglehub.dataset_download("saurabhshahane/ecommerce-text-classification")
print("Path to dataset files:", path)

path = kagglehub.dataset_download("kelixirr/arxiv-multi-label-text-classification-datasets")

print("Path to dataset files:", path)
