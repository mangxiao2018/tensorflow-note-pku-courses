import tensorflow as tf
import numpy as np

a = np.arange(0, 5)
# 把numpy数组转成张量
b = tf.convert_to_tensor(a, dtype=tf.int64)

print("a:",a)
print("b:",b)