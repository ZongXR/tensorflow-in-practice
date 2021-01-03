#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import Callback
import numpy as np
from IPython import get_ipython


class MyCallback(Callback):
    """
    自定义回调\n
    """
    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        每轮结束后回调\n
        :param epoch: 轮数
        :param logs: 日志
        :return: 空
        """
        if logs["accuracy"] > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


def train_mnist():
    """
    训练模型\n
    :return: (轮数, 精度)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # YOUR CODE SHOULD START HERE
    x_train = x_train / 255
    x_test = x_test / 255
    # YOUR CODE SHOULD END HERE
    model = Sequential([
        # YOUR CODE SHOULD START HERE
        Flatten(input_shape=x_train.shape[1:]),
        Dense(units=512, activation=tf.nn.relu),
        Dense(units=len(np.unique(y_train)), activation=tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # model fitting
    history = model.fit(
        # YOUR CODE SHOULD START HERE
        x=x_train,
        y=y_train,
        epochs=10,
        callbacks=[MyCallback()]
        # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 训练模型
    train_mnist()
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

