import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

# MNIST-Datensatz laden
(x_train, _), (x_test, _) = mnist.load_data()

# Daten normalisieren und in Vektoren umwandeln
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# Dimension der latenten Repräsentation
encoding_dim = 32  # Anzahl der Neuronen im Engpass (Bottleneck)

# Encoder definieren
tinput = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(tinput)

# Decoder definieren
decoded = Dense(784, activation='sigmoid')(encoded)

# Autoencoder-Modell erstellen
autoencoder = Model(tinput, decoded)

# Encoder-Modell (zum Extrahieren von Merkmalen)
encoder = Model(tinput, encoded)



# Autoencoder kompilieren
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

# Modell trainieren
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Testbilder codieren und decodieren
encoded_imgs = encoder.predict(x_test)

# Zeige die Bottleneck-Daten an
for i in range(10):
    plt.figure(figsize=(10, 1))
    plt.title(f'Bottleneck-Repräsentation für Bild {i}')
    plt.imshow(encoded_imgs[i].reshape(1, -1), cmap='viridis', aspect='auto')
    plt.colorbar(label='Aktivierungswerte')
    plt.show()

decoded_imgs = autoencoder.predict(x_test)

# Ergebnisse visualisieren
n = 10  # Anzahl der Bilder
plt.figure(figsize=(20, 4))
for i in range(n):
    # Originalbilder
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Rekonstruierte Bilder
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()