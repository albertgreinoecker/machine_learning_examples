import time

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras


def predict_image(path):
    image = Image.open(path)

    image = image.resize(image_size)

    im_arr = np.array(image)
    im_arr = np.reshape(image, reshape)
    print(im_arr.shape)
    pred = model.predict(im_arr)
    print(pred)


model = keras.models.load_model("/home/albert/model.mdl.keras")
model.load_weights("/home/albert/model.weights.h5")

model.summary()

dim = 128
input_shape = (dim, dim, 3)
image_size = (dim, dim)
reshape = (1, dim, dim, 3)


predict_image("/home/albert/tmp/img/stone/stone_9_albertgreinoecker.png")
predict_image("/home/albert/tmp/img/paper/paper_2_albertgreinoecker.png")
predict_image("/home/albert/tmp/img/scissor/scissor_13_albertgreinoecker.png")

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
img_counter = 0
frame_set = []
start_time = time.time()


categories = ["paper", "scissor", "stone"]


howoften = 1

while True:
    ret, frame = capture.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if time.time() - start_time >= howoften:  # Check if howoften sec passed
        start_time = time.time()
        image = Image.fromarray(frame)
        image = image.resize(image_size)

        # im_arr = np.array(image)
        # im_arr = np.reshape(image, reshape)

        im_arr = np.array(image).astype("float32") / 255.0  # wichtig: Normalisieren!
        im_arr = np.reshape(im_arr, reshape)

        print(im_arr.shape)
        pred = model.predict(im_arr)
        print(pred)
        cat = pred.argmax()
        print("Predicted category:", categories[cat])
