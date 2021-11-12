import tensorflow as tf

x = tf.Variable(4)
# 4 - 1 = 3 使用assign_sub进行自减1操作
x.assign_sub(1)
print("x:", x)