import tensorflow as tf

# one_hot 独热码，与设置的depth分类数有关，输出时设置的几分类，对某个数进行独热码转换时就用几个0/1来表示
# 如3分类，那对于1，0，2各用三个0/1组成来表示
# 1：[0. 1. 0.]
# 0：[1. 0. 0.]
# 2：[0. 0. 1.]
classes = 3
labels = tf.constant([1, 0, 2])
output = tf.one_hot(labels, depth=classes)
print("result of labeles:", output)
print("\n")