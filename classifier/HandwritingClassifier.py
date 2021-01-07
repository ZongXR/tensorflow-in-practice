#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard, History
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from typing import Union
from IPython import get_ipython


class HandwritingClassifier:
    _model: Sequential
    _SIZE = (28, 28, 1)
    _model_weights: str
    _logs: str

    def __init__(self, _model_weights: str, _logs: str) -> None:
        """
        构造方法\n
        :param _model_weights: 模型权重路径
        :param _logs: 日志路径
        """
        super().__init__()
        self._model_weights = _model_weights
        self._logs = _logs

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 编译好的模型
        """
        self._model = Sequential([
            # YOUR CODE STARTS HERE
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", data_format="channels_last", name="con2d_1", input_shape=self._SIZE),
            MaxPool2D(pool_size=(2, 2), name="pool2d_1"),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", data_format="channels_last", name="con2d_2"),
            MaxPool2D(pool_size=(2, 2), name="pool2d_2"),
            Flatten(data_format="channels_last", name="flatten_3"),
            Dense(units=128, activation="relu", name="dense_3"),
            Dense(units=10, activation="softmax", name="dense_4")
            # YOUR CODE ENDS HERE
        ])
        self._model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self._model

    def fit(self, _x_train: np.ndarray, _y_train: np.ndarray) -> Union[History, None]:
        """
        训练模型\n
        :param _x_train: x样本
        :param _y_train: y样本
        :return: 历史记录
        """
        try:
            self._model.load_weights(self._model_weights)
            return None
        except OSError:
            _history = self._model.fit(
                x=_x_train,
                y=_y_train,
                epochs=20,
                workers=-1,
                use_multiprocessing=True,
                callbacks=[
                    EarlyStopByAccuracy(0.998),
                    TensorBoard(log_dir=self._logs)
                ]
            )
            return _history

    def save_weights(self, _model_weights: str = None) -> None:
        """
        保存权重\n
        :param _model_weights: 保存路径
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)

    @classmethod
    def pre_process(cls, *_datas: np.ndarray) -> (np.ndarray, ):
        """
        数据预处理\n
        :param _datas:图像
        :return:处理后的图像
        """
        result = []
        for _data in _datas:
            result.append(np.expand_dims(_data / 255, 3))
        return tuple(result)


if __name__ == '__main__':
    # 设置显存自动增长
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    # 获取样本
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 构建模型
    clf = HandwritingClassifier("../models/HandwritingClassifier.h5", "../logs/HandwritingClassifier")
    clf.build_model()
    x_train, x_test = HandwritingClassifier.pre_process(x_train, x_test)
    clf.fit(x_train, y_train)
    clf.save_weights()
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')


