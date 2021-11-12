import tensorflow as tf
import numpy as np
## error
def normalize(data):
    x_data = data.T
    for i in range(4):
        x_data[i] = (x_data[i] - tf.reduce_min(x_data[i]))/ (tf.reduce_max(x_data[i]) - tf.reduce_min(x_data[i]))
    return x_data.T

data = np.array([[1,2,3],[2,3,4],[3,4,5]])
data = tf.convert_to_tensor(data, dtype=tf.int32)
x_data = normalize(data)
print(x_data)