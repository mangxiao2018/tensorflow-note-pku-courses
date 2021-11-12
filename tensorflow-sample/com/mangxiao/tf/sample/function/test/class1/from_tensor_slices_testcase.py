import tensorflow as tf

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
# 把特征值与标签值一一配对
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

for e in dataset:
    print(e)