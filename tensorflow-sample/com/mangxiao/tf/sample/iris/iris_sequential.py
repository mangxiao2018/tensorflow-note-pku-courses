# 1、引入相关模块
import tensorflow as tf
from sklearn import datasets
import numpy as np

# 2、加载数据集
feature_train = datasets.load_iris().data
label_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(feature_train)
np.random.seed(116)
np.random.shuffle(label_train)
tf.random.set_seed(116)

# 3、搭建网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])
# 4、配置训练方法
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# 5、执行训练过程
model.fit(feature_train, label_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# 6、打印网络结构及参数统计信息
model.summary()
