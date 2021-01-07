#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History
from typing import Union
import numpy as np


class LinearRegressor:
    _model: Sequential
    _model_weights: str

    def __init__(self, _model_weights: str) -> None:
        """
        构造方法\n
        :param _model_weights: 模型权重
        """
        super().__init__()
        self._model_weights = _model_weights

    def build_model(self, input_shape: (float, )) -> Sequential:
        """
        构建模型\n
        :param input_shape: 一个样本的大小
        :return: 编译好的模型
        """
        self._model = Sequential([
            Dense(units=1, activation="linear", name="dense_1", input_shape=input_shape)
        ])
        self._model.compile(
            optimizer="sgd",
            loss="mean_squared_error",
            metrics=["mean_squared_error"]
        )
        return self._model

    def fit(self, _x_train: np.ndarray, _y_train: np.ndarray, _epochs: int = 500, _batch_size: int = None) -> Union[History, None]:
        """
        训练\n
        :param _x_train: 训练集x
        :param _y_train: 训练集y
        :param _epochs: 轮数
        :param _batch_size: 批量大小
        :return: 训练历史数据
        """
        try:
            self._model.load_weights(self._model_weights)
            return None
        except OSError:
            _history = self._model.fit(
                x=_x_train,
                y=_y_train,
                epochs=_epochs,
                batch_size=_batch_size,
                workers=-1,
                use_multiprocessing=True
            )
            return _history

    def predict(self, _x_test: np.ndarray) -> float:
        """
        预测一批样本\n
        :param _x_test: 测试集x
        :return: 预测结果
        """
        return self._model.predict(_x_test)

    def predict_one(self, _x_test: [float]) -> float:
        """
        预测一个样本\n
        :param _x_test: 测试集x
        :return: 预测结果
        """
        return self._model.predict(np.array([_x_test]))

    def save_weights(self, _model_weights: str) -> None:
        """
        保存权重\n
        :param _model_weights: 权重保存路径
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)




