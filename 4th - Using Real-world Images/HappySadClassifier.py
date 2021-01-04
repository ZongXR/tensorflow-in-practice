#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from IPython import get_ipython
import os


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
        if logs["accuracy"] > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


def train_happy_sad_model():
    model = Sequential([
        # Your Code Here
        Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(150, 150, 3), name="conv2d_1"),
        MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_1"),
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
        MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_2"),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv2d_3"),
        MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_3"),
        Flatten(data_format="channels_last", name="flatten_4"),
        Dense(units=512, activation="relu", name="dense_4"),
        Dense(units=1, activation="sigmoid", name="dense_5")
    ])

    model.compile(
        optimizer=RMSprop(lr=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    train_generator = ImageDataGenerator(rescale=1/255).flow_from_directory(
        directory="../data/HappySad",
        target_size=(150, 150),
        class_mode="binary",
        batch_size=32
    )
    # model fitting
    history = model.fit_generator(
        generator=train_generator,
        epochs=20,
        workers=-1,
        use_multiprocessing=True,
        callbacks=[MyCallback()]
    )
    # model fitting
    return history.history['accuracy'][-1]


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 训练模型
    train_happy_sad_model()
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

