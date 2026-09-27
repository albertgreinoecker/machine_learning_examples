import json
import os
import urllib

import numpy as np
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Do not use the GPU


# Download mapping of ImageNet class indices to labels
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
class_index = json.load(urllib.request.urlopen(url))

# Example: print first 10 categories
for k in range(1000):
    print(k, class_index[str(k)])


model = EfficientNetB0(weights="imagenet")
model.summary()


# Bild laden
img_path = "data/cat.png"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# Verwende das gleiche Bild wie oben
# 3. Vorhersage
preds = model.predict(x)
print("Top-5 Predictions:")
print(decode_predictions(preds, top=5)[0])
