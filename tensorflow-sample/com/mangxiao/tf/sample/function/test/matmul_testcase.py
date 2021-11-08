import tensorflow as tf

a = tf.ones([3, 2])
b = tf.fill([2, 3], 3.)

print("a:", a)
print("b:", b)
# 矩阵a与b相乘
print("a*b:", tf.matmul(a, b))