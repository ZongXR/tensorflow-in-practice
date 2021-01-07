#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from callbacks.MyCallback import MyCallBack
from IPython import get_ipython


class HappySadClassifier:
    """
    高兴悲伤分类器\n
    """
    _model: Sequential
    _SIZE = (150, 150, 3)
    _model_path: str
    _logs: str

    def __init__(self, _model_path: str, _logs: str) -> None:
        """
        构造方法\n
        :param _model_path 模型权重路径
        :param _logs 日志路径
        """
        super().__init__()
        self._model_path = _model_path
        self._logs = _logs

    def build_model(self):
        self._model = Sequential([
            # Your Code Here
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=HappySadClassifier._SIZE, name="conv2d_1"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_1"),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_2"),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv2d_3"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_3"),
            Flatten(data_format="channels_last", name="flatten_4"),
            Dense(units=512, activation="relu", name="dense_4"),
            Dense(units=1, activation="sigmoid", name="dense_5")
        ])
        self._model.compile(
            optimizer=RMSprop(lr=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return self._model

    def fit(self, _data: str) -> float:
        """
        训练模型\n
        :param: _data: 数据目录
        :return: 加载权重，返回空；训练模型，返回最后一轮精度
        """
        try:
            self._model.load_weights(self._model_path)
        except OSError:
            train_generator = ImageDataGenerator(rescale=1/255).flow_from_directory(
                directory=_data + "/train",
                target_size=HappySadClassifier._SIZE[:-1],
                class_mode="binary",
                batch_size=32
            )
            logs = TensorBoard(self._logs)
            history = self._model.fit_generator(
                generator=train_generator,
                epochs=20,
                workers=-1,
                use_multiprocessing=True,
                callbacks=[MyCallBack(), logs]
            )
            # model fitting
            return history.history['accuracy'][-1]

    def save_weights(self) -> None:
        """
        保存权重\n
        :return: 空
        """
        self._model.save_weights(self._model_path)


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 训练模型
    clf = HappySadClassifier("../models/HappySadClassifier.h5", "../logs/HappySadClassifier")
    clf.build_model()
    clf.fit("../data/HappySad")
    clf.save_weights()
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

