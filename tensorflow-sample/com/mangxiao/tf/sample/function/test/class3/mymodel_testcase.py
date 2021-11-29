import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
np.random.set_seed(116)

class IrisMode(Model):
    def __init__(self):
        super(IrisMode, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
    def call(self, x):
        y = self.d1(x)
        return y

model = IrisMode()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()