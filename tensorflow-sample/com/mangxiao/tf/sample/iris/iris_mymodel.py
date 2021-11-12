import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

feature_train = datasets.load_iris().data
label_train = datasets.load_iris().target
# print(feature_train[0])

np.random.seed(116)
np.random.shuffle(feature_train)
np.random.seed(116)
np.random.shuffle(label_train)
tf.random.set_seed(116)

# 定义网络模型
class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)
        return y

def plot_loss(results):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    plt.show()

# 初始化模型对象
model = IrisModel()

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# 训练
results =model.fit(feature_train, label_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
# 概要
model.summary()

plot_loss(results)