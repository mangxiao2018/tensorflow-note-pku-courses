import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))
    # y = x^2
    y = tf.pow(x, 2)
    print("x:", x)
    print("y:", y)
    # y = 2x,x=3.0
grad = tape.gradient(y, x)
print("grad:", grad)