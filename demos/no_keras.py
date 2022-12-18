# -*- coding: utf-8 -*-
"""
完全不使用keras高阶API
"""
from typing import Tuple
import tensorflow as tf
from tensorflow import Module, Tensor, Variable, GradientTape, random_normal_initializer
from tensorflow.keras.datasets.boston_housing import load_data


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.reset_default_graph()


class LinearRegression(Module):

    def __init__(self, input_shape: Tuple[int, int], name=None):
        super(LinearRegression, self).__init__(name=name)
        with self.name_scope:  # 相当于with tf.name_scope("demo_module")
            self.w = Variable(random_normal_initializer(mean=0.0, stddev=1.0)(shape=(input_shape[1], 1), dtype=tf.float64), trainable=True)
            self.b = Variable(random_normal_initializer(mean=0.0, stddev=1.0)(shape=(input_shape[0], 1), dtype=tf.float64), trainable=True)

    @tf.function
    def __call__(self, x: Tensor) -> Tensor:
        with self.name_scope:
            y = x @ self.w + self.b
            return y


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    model = LinearRegression(input_shape=(404, 13), name="linear")
    optimizer = tf.optimizers.SGD(learning_rate=1e-6)

    for epoch in range(500):
        with GradientTape() as tape:
            y_pred = model(x_train)
            loss = tf.losses.MeanSquaredError()(y_train, y_pred)
        tf.print("--------------epoch: %d, loss: %f --------------" % (epoch, loss.numpy()))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

