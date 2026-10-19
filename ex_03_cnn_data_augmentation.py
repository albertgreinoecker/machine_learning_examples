import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.datasets import mnist
from tensorflow import keras

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

"""
 the data, split between train and test sets
x_train: images for training
y_train: labels for training
x_test: images for testing the model
y_test: labels for testing the model
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
# Scale images to the [0, 1] range
# Cast to float values before to make sure result ist float
"""
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# augmentation layers expect a channel dimension: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors (the labels) to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_labels = y_test  # use this to leave the labels untouched
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Data augmentation
Each training image is randomly modified in every epoch, so the network
(almost) never sees exactly the same image twice. This reduces overfitting
and makes the model more robust against slightly rotated, shifted or
scaled digits.

IMPORTANT: these layers are only active during training (model.fit).
During evaluate/predict they simply pass the images through unchanged.
"""
data_augmentation = keras.Sequential(
    [
        # rotate by up to +/- 10% of a full circle (= +/- 36 degrees)
        layers.RandomRotation(factor=0.1, fill_mode="constant", fill_value=0.0),
        # shift by up to +/- 10% of the image height / width
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=0.0),
        # zoom in or out by up to 10%
        layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=0.0),
    ],
    name="data_augmentation",
)

"""
## Show some augmented images
The first column shows the original image, the other columns show
random augmentations of the same image (training=True forces augmentation).
"""
num_examples = 5
num_variants = 6
fig, axes = plt.subplots(num_examples, num_variants, figsize=(num_variants * 1.5, num_examples * 1.5))
for row in range(num_examples):
    image = x_train[row : row + 1]  # keep the batch dimension
    for col in range(num_variants):
        shown = image if col == 0 else data_augmentation(image, training=True)
        axes[row, col].imshow(np.squeeze(shown), cmap="gray")
        axes[row, col].axis("off")
    axes[row, 0].set_title("original", fontsize=8)
plt.tight_layout()
plt.show()

"""
## Build the model
The augmentation block is simply the first part of the model.
"""
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 30

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# draw the learn function
# note: training accuracy is often lower than validation accuracy here,
# because the training images are harder (augmented) than the validation images
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
## Do some predictions on the test dataset and compare the results
"""

pred = model.predict(x_test)

for i in range(0, 100):
    pred_i = np.argmax(pred[i])  # get the position of the highest value within the list
    print(y_labels[i], pred_i)
