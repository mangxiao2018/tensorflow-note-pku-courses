# _*_ coding: UTF-8 _*_

# 导入模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据
feature = datasets.load_iris().data
label =  datasets.load_iris().target

# 混淆数据
np.random.seed(116)
np.random.shuffle(feature)
np.random.seed(116)
np.random.shuffle(label)
# 设置全局随机种子，跨会话
tf.random.set_seed(116)

# 切割训练集和测试集
# [:-30]表示从0到倒数第30个数，如100个数，就是1-70个数
# [-30:0]表示倒数第30个数到最后一个数，如100个数，就是70-100个数
feature_train = feature[:-30]
label_train = label[:-30]
feature_test = feature[-30:]
label_test = label[-30:]

# 数据类型转换
feature_train = tf.cast(feature_train, tf.float32)
feature_test = tf.cast(feature_test, tf.float32)

# 特征与标签值配对
train_db = tf.data.Dataset.from_tensor_slices((feature_train, label_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((feature_test, label_test)).batch(32)
'''
for e in train_db:
    print(e)
    print("-----------------------------------")
'''
# 标记可训练数据,构建神经网络骨架
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))


# 设置相关参数
lr = 0.1
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0


for epoch in range(epoch):
    # 训练
    for step, (feature_train, label_train) in enumerate(train_db):
        #print("feature_train:{}".format(feature_train))
        #print("label_train:{}".format(label_train))
        with tf.GradientTape() as tape:
            # 矩阵计算: y = x * w + b
            y = tf.matmul(feature_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(label_train, depth=3)
            # loss = (y_ - y)^2 / n 因为没有设置axis维度，所以求所有值的均值，输出为一个标量值
            loss = tf.reduce_mean(tf.square(y_ - y))
            print("loss:{}".format(loss))
            loss_all += loss.numpy()
        # 对loss中的w1,b1求一次梯度
        grads = tape.gradient(loss, [w1, b1])
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        # b1 = b1 - lr * b1_grad
        b1.assign_sub(lr * grads[1])

    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0

    # 测试
    total_correct, total_number = 0, 0
    for feature_test, label_test in test_db:
        y = tf.matmul(feature_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=label_test.dtype)
        correct = tf.cast(tf.equal(pred, label_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += label_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("---------------------------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()


