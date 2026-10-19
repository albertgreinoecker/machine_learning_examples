"""
First steps with the MNIST dataset.

This script:
  1. loads the MNIST dataset (handwritten digits 0-9, 28x28 grayscale images)
     via Keras,
  2. prints some basic information about the data (number of samples,
     array shape, raw pixel values and label of a single image),
  3. displays a single image with matplotlib,
  4. exports the first 100 training images as JPEG files. The file name
     encodes the index and the true label: "<index>_<label>.jpeg".

Note: fashion_mnist is imported as well; replace ``mnist`` with
``fashion_mnist`` in the load call (and adjust OUTPUT_DIR) to work with the
clothing dataset instead.
"""

import os

# Random integer function, used for optionally picking a random image
from random import randint  # noqa: F401

# Plotting library for visualizing images
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist  # noqa: F401

# Pillow image library, used to save numpy arrays as image files
from PIL import Image

# Load the dataset. Returns numpy arrays:
#   train_images: (60000, 28, 28) uint8, pixel values 0-255
#   train_labels: (60000,)        uint8, digit class 0-9
#   test_images:  (10000, 28, 28), test_labels: (10000,)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Target directory for the exported JPEG files
OUTPUT_DIR = "/home/albert/tmp/fashion_mnist"

# Get to know the dataset
print("trainingsdaten:")
print(len(train_images))  # number of training samples
print(len(test_images))  # number of test samples
print(train_images.shape)  # (samples, height, width)
img_no = 100  # index of the image to inspect
print(train_images[img_no])  # raw 28x28 pixel matrix
img100 = train_images[img_no]
print(train_labels[img_no])  # true label (digit) of that image

# Show the image
plt.figure()
# Alternative: show a random training image instead of a fixed one
# plt.imshow(train_images[randint(0, len(train_images) - 1)])
plt.imshow(train_images[img_no])
plt.grid(False)
# Non-blocking so the script continues while the window is open
plt.show(block=False)

# Export the first 100 training images as JPEG files named "<index>_<label>.jpeg".
# Create the target directory if it does not exist yet.
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i in range(0, 100):
    im = Image.fromarray(train_images[i])  # convert 28x28 uint8 array to a grayscale image
    real = train_labels[i]  # true label, encoded in the file name
    im.save(os.path.join(OUTPUT_DIR, "%d_%d.jpeg" % (i, real)))
