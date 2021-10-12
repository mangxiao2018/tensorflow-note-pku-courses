import tensorflow as tf
import numpy as np

def normalize(data):
    x_data = data.T
    for i in range(4):
        x_data[i] = (x_data[i] - tf.reduce_min(x_data[i]))/ (tf.reduce_max(x_data[i]) - tf.reduce_min(x_data[i]))
    return x_data.T

