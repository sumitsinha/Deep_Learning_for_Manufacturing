import tensorflow as tf
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)