#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from IPython import get_ipython


class MyCallback(Callback):
    """
    自定义回调\n
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        early stop\n
        :param epoch: 轮数
        :param logs: 日志
        :return: 空
        """
        if logs["accuracy"] > 0.998:
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


def pre_process(*_datas: np.ndarray) -> (np.ndarray,):
    """
    数据预处理\n
    :param _datas: 预处理前的数据
    :return: 预处理后的数据
    """
    result = []
    for _data in _datas:
        result.append(np.expand_dims(_data / 255, 3))
    return tuple(result)


def train_mnist_conv():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # YOUR CODE STARTS HERE
    x_train, x_test = pre_process(x_train, x_test)
    # YOUR CODE ENDS HERE

    model = Sequential([
        # YOUR CODE STARTS HERE
        Conv2D(filters=64, kernel_size=(3, 3), input_shape=x_train.shape[1:], activation="relu", data_format="channels_last"),
        MaxPool2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Flatten(data_format="channels_last"),
        Dense(units=128, activation="relu"),
        Dense(units=len(np.unique(y_test)), activation="softmax")
        # YOUR CODE ENDS HERE
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        x=x_train,
        y=y_train,
        epochs=20,
        callbacks=[MyCallback()]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]


if __name__ == '__main__':
    # 设置显存自动增长
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    # 训练模型
    _, _ = train_mnist_conv()
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

