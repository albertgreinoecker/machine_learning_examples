import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
import matplotlib.pyplot as plt


# Generator-Modell erstellen
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100))  # Eingabe-Rauschvektor (100 Dimensionen)
    model.add(LeakyReLU(alpha=0.2)) #Verhindert "tote" Neuronen, indem Wert nicht 0 werden kann
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28 * 28, activation='tanh'))  # Ausgabe in Form eines 28x28 Bildes
    model.add(Reshape((28, 28)))
    return model


# Discriminator-Modell erstellen
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))  # Klassifikation: Echt oder Fake
    return model


# GAN-Modell zusammenstellen
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False  # Discriminator während GAN-Training einfrieren
    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# Training des GANs
def train_gan(generator, discriminator, gan, epochs=10000, batch_size=64, save_interval=1000):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalisierung auf [-1, 1]

    for epoch in range(epochs):
        # Zufällige echte Bilder
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        real_labels = np.ones((batch_size, 1))

        # Generierte Fake-Bilder
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))

        # Discriminator-Training
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generator-Training über GAN (Discriminator eingefroren)
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_labels = np.ones((batch_size, 1))  # Möchte den Discriminator überlisten
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Fortschritt ausgeben
        if epoch % save_interval == 0:
            print(f"Epoch {epoch}: [D loss: {d_loss[0]}] [G loss: {g_loss}]")
            plot_generated_images(generator, epoch)


# Generierte Bilder anzeigen
def plot_generated_images(generator, epoch, examples=10):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)

    plt.figure(figsize=(10, 2))
    for i in range(examples):
        plt.subplot(1, examples, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.show()


# Modelle erstellen
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# GAN trainieren
train_gan(generator, discriminator, gan, epochs=10000, batch_size=64, save_interval=2000)
