import os

import keras
import numpy as np
from keras import layers
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Do not use the GPU

# Transfer learning: reuse the pretrained ResNet50 (ImageNet) as feature extractor
# and replace the 1000-class ImageNet head with a head for our own classes.
#
# Expected folder structure (folder names = class names):
#   data_dir/
#       class_a/  *.jpg
#       class_b/  *.jpg
#       ...
# Set DATA_DIR to your own image folder. If it is None, the TensorFlow flowers
# dataset (5 classes) is downloaded as an example.
DATA_DIR = None

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
FINE_TUNE_EPOCHS = 3

if DATA_DIR is None:
    path = keras.utils.get_file(
        "flower_photos",
        origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        extract=True,
    )
    # Keras 3 extracts into "<name>_extracted/flower_photos"
    DATA_DIR = os.path.join(path, "flower_photos") if os.path.isdir(os.path.join(path, "flower_photos")) else path

###########################################
# Load data

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)

###########################################
# Build model

# Pretrained ResNet50 without the ImageNet classification head (include_top=False)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False  # freeze pretrained weights

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Custom classification head for our own classes
model = keras.Sequential(
    [
        keras.Input(shape=IMG_SIZE + (3,)),
        data_augmentation,
        layers.Lambda(preprocess_input),  # ResNet50-specific preprocessing (BGR, mean subtraction)
        # --- Feature Extraction (pretrained) ---
        base_model,
        # --- Classification head ---
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

###########################################
# Phase 1: train only the new head

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

###########################################
# Phase 2 (optional): fine-tune the upper layers of ResNet50

base_model.trainable = True
for layer in base_model.layers[:-30]:  # unfreeze only the last 30 layers
    layer.trainable = False
# Keep BatchNorm layers frozen: in a Sequential model we cannot call base_model(x, training=False),
# and trainable BatchNorm layers would update their statistics and break the pretrained features
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Recompile with a very small learning rate to avoid destroying the pretrained weights
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS)

###########################################
# Prediction with our own class names
# (decode_predictions only works for the 1000 ImageNet classes!)

images, labels = next(iter(val_ds))
preds = model.predict(images)

for i in range(5):
    pred_idx = np.argmax(preds[i])
    print(
        f"True: {class_names[labels[i]]:12s} Predicted: {class_names[pred_idx]:12s} ({preds[i][pred_idx]:.2f})"
    )

# Prediction for a single image file
img = keras.utils.load_img("data/cat.png", target_size=IMG_SIZE)
x = np.expand_dims(keras.utils.img_to_array(img), axis=0)  # no preprocess_input: it is part of the model
pred = model.predict(x)[0]
print("data/cat.png ->", class_names[np.argmax(pred)], f"({pred.max():.2f})")

model.save("resnet_transfer.keras")

# Loading the model again requires the function used in the Lambda layer:
# model = keras.models.load_model(
#     "resnet_transfer.keras", safe_mode=False, custom_objects={"preprocess_input": preprocess_input}
# )
