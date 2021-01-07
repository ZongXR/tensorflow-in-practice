#!/usr/bin/env python
# coding: utf-8


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.callbacks import History, TensorBoard
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy


class FashionClassifier:
    _model: Sequential
    _SIZE = (28, 28, 1)
    _model_weights: str
    _logs: str

    def __init__(self, _model_weights: str, _logs: str) -> None:
        """
        构造方法\n
        :param _model_weights: 权重保存路径
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
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv2d_1", input_shape=FashionClassifier._SIZE),
            MaxPool2D(pool_size=(2, 2), name="pool2d_1"),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            MaxPool2D(pool_size=(2, 2), name="pool2d_2"),
            Flatten(name="flatten_3"),
            Dense(units=128, activation="relu", name="dense_3"),
            Dense(units=10, activation="softmax", name="softmax_4")
        ])
        self._model.compile(
            optimizer="Adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self._model.summary()
        return self._model

    def fit(self, _x_train: np.ndarray, _y_train: np.ndarray) -> Union[History, None]:
        """
        训练模型\n
        :param _x_train: 训练集x
        :param _y_train: 训练集y
        :return: 历史数据
        """
        try:
            self._model.load_weights(self._model_weights)
            return None
        except OSError:
            _history = self._model.fit(
                x=_x_train,
                y=_y_train,
                epochs=10,
                workers=-1,
                use_multiprocessing=True,
                callbacks=[
                    EarlyStopByAccuracy(0.95),
                    TensorBoard(self._logs)
                ]
            )
            return _history

    def save_weights(self, _model_weights: str = None) -> None:
        """
        保存权重\n
        :param _model_weights: 权重保存路径
        :return:
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)

    def evaluate(self, _x_test: np.ndarray, _y_test: np.ndarray) -> [float]:
        """
        评价模型\n
        :param _x_test: 测试集x
        :param _y_test: 测试集
        :return: loss, accuracy
        """
        self._model.evaluate(x=_x_test, y=_y_test)

    @classmethod
    def pre_process(cls, *_datas: np.ndarray) -> (np.ndarray, ):
        """
        数据预处理\n
        :param _datas: 数据
        :return: 图像
        """
        result = []
        for _data in _datas:
            _data = np.expand_dims(_data, 3)
            result.append(_data / 255)
        return tuple(result)

    def show_inner(self, _x: np.ndarray, _idxs: (int, ), _filename: str) -> None:
        """
        展示神经网络内部细节\n
        :param _x: 样本图像
        :param _idxs: 第几个样本
        :param _filename: 保存图片的位置
        :return: 空
        """
        plt.figure(1)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.subplots(len(_idxs), 4)
        channel_index = 1
        outputs = [x.output for x in self._model.layers]
        activation_model = Model(inputs=self._model.inputs, outputs=outputs)
        for i in range(0, 4):
            output = activation_model.predict(_x[_idxs, :, :, :])[i]
            for k, img in enumerate(output):
                plt.subplot(len(_idxs), 4, k * 4 + i + 1)
                plt.imshow(img[:, :, channel_index], cmap='inferno')
                plt.title("样本%d的%s层" % (_idxs[k], self._model.layers[i].name))
        plt.tight_layout()
        plt.savefig(_filename)
        plt.show()

