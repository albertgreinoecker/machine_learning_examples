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

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 3
input_shape = (180, 180, 3)

d = keras.preprocessing.image_dataset_from_directory('/home/albert/tmp/img', image_size=(180, 180), label_mode='categorical', batch_size= 1000)

images = None
labels = None

print("Class names: %s" % d.class_names) # Welche Kategorien gibt es generell

for d, l in d.take(1):
    images = d
    labels = l

print(images.shape)
print(labels.shape)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
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


# ## Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
history = model.fit(images, labels, batch_size=128, epochs=1, validation_split=0.1)

image = Image.open('/home/albert/tmp/img/papier/stone_9_albertgreinoecker.png')
newsize = (180, 180)
image = image.resize(newsize)

im_arr = np.array(image)
im_arr = np.reshape(image, (1, 180, 180, 3))
print(im_arr.shape)
pred = model.predict(im_arr)
print(pred)

# """
# How to load and save the model
# """
#
# model.save('/home/albert/model.mdl')
# model.save_weights("/home/albert/model.h5")
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