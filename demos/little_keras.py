# -*- coding: utf-8 -*-
"""
使用一部分keras高阶API
"""
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.datasets.boston_housing import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import clear_session


clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    model = Sequential([
        Dense(units=1, activation=None, input_shape=(x_train.shape[1], ))
    ])
    optimizer = SGD(learning_rate=1e-6)

    for epoch in range(500):
        with GradientTape() as tape:
            y_pred = model(x_train)
            loss = MeanSquaredError()(y_train, y_pred)
        tf.print("--------------epoch: %d, loss: %f --------------" % (epoch, loss.numpy()))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

