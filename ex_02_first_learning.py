import pandas as pd
import collections #used for counting items of a list
import os
import matplotlib.pyplot as plt
import json

from keras.src.datasets import mnist
import keras
import numpy as np
from keras import layers



os.environ["KERAS_BACKEND"] = "tensorflow"
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
#download the images
"""
path = "/home/albert/tmp/keras1/"

print ("storing images.....")
# for i in range(1,20):
#
#     plt.imshow(x_train[i])
#     #plt.show()
#     plt.savefig(path + str(i) +   " .png")



"""
# Scale images to the [0, 1] range
# Cast to float values before to make sure result ist float
"""
x_train = x_train.astype("float32") / 255
print(x_train.shape, "train samples")
x_test = x_test.astype("float32") / 255



# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape, "x_train shape:")
print(x_train.shape[0], "number of train samples")
print(x_test.shape[0], "number of test samples")

nr_labels_y = collections.Counter(y_train) #count the number of labels
print(nr_labels_y, "Number of labels")

# convert class vectors (the labels) to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

print(y_train[10])
y_labels = y_test #use this to leave the labels untouched
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train[0])
"""
## Build the model
"""

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train[1][400:600])

model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 3

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#draw the learn function
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()
"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
## Do some Preodictions on the test dataset and compare the results
"""

pred = model.predict(x_test)

print(pred[1]) #Prediction for image 1
pred_1 = np.argmax(pred[1])
print(pred_1)
np.argmax(y_test[1])

for i in range(0,1000):
    pred_i = np.argmax(pred[i]) # get the position of the highest value within the list
    real = np.argmax(y_test[i])
    if  real != pred_i:
        print(real, pred_i)


"""
How to load and save the model
"""

model.save('out/ex_02_model.keras')
model.save_weights('out/ex_02.weights.h5')

weights = model.get_weights()
j =json.dumps(pd.Series(weights).to_json(orient='values'), indent=3)

model = keras.models.load_model('out/ex_02_model.keras')
model.load_weights('out/ex_02.weights.h5')

model_json = model.to_json()
print (model_json)