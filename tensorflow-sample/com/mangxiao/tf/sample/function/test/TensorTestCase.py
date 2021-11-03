import tensorflow as tf

a = tf.constant([1,3,5], dtype=tf.int64)
print(a)

b = tf.zeros([2,3], dtype=tf.int64)
print(b)

c = tf.ones([2,2,3], dtype=tf.int64)
print(c)

d = tf.fill([2,4], 8)
print(d)
print("------------------------------------------------------------------")
e = tf.random.normal([2,3], mean=0.5, stddev=1)
print(e)

f = tf.random.truncated_normal([2,3], mean=0.5, stddev=1)
print(f)



