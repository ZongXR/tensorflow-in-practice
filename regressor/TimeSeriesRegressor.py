#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History, TensorBoard
from typing import Union


class TimeSeriesRegressor:
    _model: Sequential
    _input_length: int
    _model_weights: str
    _logs: str

    def __init__(self, _input_length: int, _model_weights: str, _logs: str) -> None:
        """
        输入时间序列的长度\n
        :param _input_length: 输入时间序列的长度
        :param _model_weights: 模型权重
        :param _logs: 日志
        """
        super().__init__()
        self._input_length = _input_length
        self._model_weights = _model_weights
        self._logs = _logs

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 编译好的模型
        """
        self._model = Sequential([
            Dense(units=10, activation="relu", name="dense_1", input_shape=[self._input_length]),
            Dense(units=10, activation="relu", name="dense_2"),
            Dense(units=1, activation=None, name="dense_3")
        ])
        self._model.compile(
            optimizer=SGD(lr=5e-6, momentum=0.9),
            loss="mse",
            metrics=["mse"]
        )
        return self._model

    def fit(self, _data_train: tf.raw_ops.PrefetchDataset) -> dict:
        """
        训练模型\n
        :param _data_train: 训练数据集
        :return: 训练历史记录
        """
        _history: History = self._model.fit(
            _data_train,
            epochs=100,
            workers=-1,
            use_multiprocessing=True,
            callbacks=[
                TensorBoard(log_dir=self._logs)
            ]
        )
        return _history.history

    def predict(self, _x_test: Union[np.ndarray, tf.raw_ops.PrefetchDataset]) -> np.ndarray:
        """
        预测\n
        :param _x_test: 测试集x
        :return: 预测值
        """
        if type(_x_test) == np.ndarray:
            return self._model.predict(np.expand_dims(_x_test, 0))
        else:
            return self._model.predict(_x_test)

