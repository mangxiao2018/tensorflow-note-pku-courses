import tensorflow as tf
# N次方,平方,开方
a = tf.fill([1, 2], 4.)
print("a:", a)
print("a的N次方:", tf.pow(a, 3))
print("a的平方:", tf.square(a))
print("a的开方:", tf.sqrt(a))