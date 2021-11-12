import tensorflow as tf
import numpy as np

test = np.array([[1,2,3], [2,3,4],[5,4,3],[8,7,2]])
print("test:", test)
# 求指定行(axis=1)/列(axis=0)上的最大值索引
print("每一列的最大值的索引:", tf.argmax(test, axis=0))
print("每一行的最大值的索引:", tf.argmax(test, axis=1))