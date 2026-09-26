import numpy as np
import pandas as pd
import collections #used for counting items of a list
from tensorflow import keras
from keras import layers
from keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
from tensorflow import keras
import json
from PIL import Image
import os
import tensorflow as tf
"""
## Prepare the data
"""

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Do not use the GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
tf.config.optimizer.set_jit(False)

# Model / data parameters
dim = 128
num_classes = 3
input_shape = (dim, dim, 3)
image_size = (dim,dim)
reshape = (1, dim,dim,3)
d = keras.preprocessing.image_dataset_from_directory('/home/albert/tmp/img', image_size=image_size, label_mode='categorical', batch_size= 1000)

images = None
labels = None

print("Class names: %s" % d.class_names) # Welche Kategorien gibt es generell

for d, l in d.take(1):
    images = d
    labels = l

print(images.shape)
print(labels.shape)

model = keras.Sequential([
    keras.Input(shape=input_shape),

    # --- Feature Extraction ---
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # --- Classification head ---
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
#
history = model.fit(images, labels, batch_size=128, epochs=10, validation_split=0.1)

image = Image.open('/home/albert/tmp/img/stone/stone_9_albertgreinoecker.png')

image = image.resize(image_size)

im_arr = np.array(image)
im_arr = np.reshape(image, reshape)
print(im_arr.shape)
pred = model.predict(im_arr)
print(pred)


model.save('/home/albert/model.mdl.keras')
model.save_weights('/home/albert/model.weights.h5')
#
# weights = model.get_weights()
# j =json.dumps(pd.Series(weights).to_json(orient='values'), indent=3)
# #print(j)
#
# model = keras.models.load_model('/home/albert/model.mdl')
# model.load_weights("/home/albert/model.h5")
#
# model_json = model.to_json()
# #print (model_json)