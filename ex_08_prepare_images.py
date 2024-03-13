
from tensorflow import keras
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf

# LOAD IMAGES FROM DIRECTORY
dataset = keras.preprocessing.image_dataset_from_directory('/home/albert/tmp/img',
    image_size=(256, 256),  # Resize images to this size
    batch_size=32,  # Number of images to load at each iteration
    label_mode='categorical'  # 'categorical', 'binary', 'int', or None
)

# PRINT BASIC INFORMATION
print(dataset.class_names)
print(dataset)
for data, labels in dataset.take(1):  # Take 1 batch from the dataset
    print(f'Data shape: {data.shape}')  # Should print (batch_size, 256, 256, 3)
    print(f'Labels shape: {labels.shape}')  # Should print (batch_size,)

# Transform batch dataset to a simple PrefectchDataset
dataset = dataset.cache().shuffle(1000).prefetch(buffer_size= 32)
#dataset = np.expand_dims(dataset, -1)

# CREATE A MODEL
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    # Add more layers as needed
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # num_classes should match your dataset
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# LEARN
model.fit(dataset, epochs=1)
model.summary()

# LOAD AN IMAGE FROM FILE
img = image.load_img('/home/albert/tmp/img/stone/stone001.png', target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)  # Adjust this according to your model's requirements

img_batch = tf.expand_dims(img_array, 0) #Add a Batch Dimension

predictions = model.predict(img_batch) # Make predictions

print(predictions)
predicted_class = tf.argmax(predictions[0]).numpy()
print(predicted_class)