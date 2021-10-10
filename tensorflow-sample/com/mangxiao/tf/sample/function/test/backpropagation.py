import tensorflow as tf
# tf.Variable()让数据变成可训练的
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 学习率，是个超参
lr = 0.2
# 循环训练次数
epoch = 40

for epoch in range(epoch):
    with tf.GradientTape()  as tape:  # with结构到grads 框起了梯度下降的计算过程,此行是把tf.GradientTape()定义成一个变量别名 tape,以供下面使用
        loss = tf.square(w + 1)       # loss = (w + 1)^2
    grads = tape.gradient(loss, w)    # 对loss 表达式中的w变量求导

    w.assign_sub(lr * grads)          # w = w - lr*grads

    print("After %s epoch, W is %f , loss is %f " % (epoch, w.numpy(), loss))
