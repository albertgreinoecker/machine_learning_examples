from keras.datasets import mnist, fashion_mnist
# Bibliothek für grafische Darstellung laden
import matplotlib.pyplot as plt
from PIL import Image #Image library Pillow
# Funktion für zufällige Bildauswahl laden
from random import randint

# Datensätze laden
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Kennenlernen er Datensätze
print ("trainingsdaten:")
print (len(train_images))
print (len(test_images))
print(train_images.shape)
img_no = 101
print (train_images[img_no])
img100 = train_images[img_no]
print (train_labels[img_no])
# Bild zeigen
plt.figure()
#plt.imshow(train_images[randint(1, len(train_images) - 1)])
plt.imshow(train_images[img_no])
plt.grid(False)
plt.show(block=False)

for i in range(0,100):
    im = Image.fromarray(train_images[i])
    real = train_labels[i]
    im.save("/home/albert/tmp/fashion_mnist/%d_%d.jpeg" % (i, real))