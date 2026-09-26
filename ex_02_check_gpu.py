import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # sollte []

print("VERSION:", tf.__version__)
print(tf.test.is_gpu_available())  # für ältere TF-Versionen
print(tf.test.is_built_with_cuda())