import tensorflow as tf

a = tf.constant([1., 2., 3.], dtype=tf.float64)
print("a:", a)
# 转float64位的a为int32位的
b = tf.cast(a, tf.int32)
print("b:", b)
# 计算一个张量各维度上元素的最小值、最大值
print("minium of a:", tf.reduce_min(a))
print("maxmum of a:", tf.reduce_max(a))