import tensorflow as tf
# 创建一个2*2矩阵，数值介于0-1之间，符合均匀分布
f = tf.random.uniform([2, 2], minval=0, maxval=1)

print("f:", f)
