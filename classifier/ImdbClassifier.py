#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy


class ImdbClassifier:
    _self: Sequential
    _model_weights: str
    _logs: str
    _token: Tokenizer
    _model: Sequential

    def __init__(self, _model_weights: str, _logs: str) -> None:
        """
        构造方法\n
        :param _model_weights: 模型权重保存路径
        :param _logs: 日志保存路径
        """
        super().__init__()
        self._model_weights = _model_weights
        self._logs = _logs
        self._token = Tokenizer(
            num_words=10000,
            oov_token="<OOV>"
        )

    def pre_process(self, _x: [str], _y: [int]) -> (np.ndarray, ):
        """
        数据预处理\n
        :param _x: 文本列表
        :param _y: 标签列表
        :return: 文本序列, 标签序列
        """
        return pad_sequences(
            sequences=self._token.texts_to_sequences(_x),
            maxlen=120,
            padding="pre",
            truncating="post"
        ), np.array(_y)

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 编译好的模型
        """
        self._model = Sequential([
            Embedding(input_dim=10000, output_dim=16, name="embedding_1", input_length=120),
            Flatten(name="flatten_2"),
            Dense(units=6, activation="relu", name="dense_2"),
            Dense(units=1, activation="sigmoid", name="dense_3")
        ])
        self._model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        self._model.summary()
        return self._model

    def fit(self, _x_train: [str], _y_train: [int], _x_validation: [str], _y_validation: [int]) -> dict:
        """
        训练模型\n
        :param _x_train: x训练集
        :param _y_train: y训练集
        :param _x_validation: x验证集
        :param _y_validation: y验证集
        :return: 训练历史记录
        """
        self._token.fit_on_texts(_x_train)
        _x, _y = self.pre_process(_x_train, _y_train)
        _x_val, _y_val = self.pre_process(_x_validation, _y_validation)
        history = self._model.fit(
            x=_x,
            y=_y,
            validation_data=(_x_val, _y_val),
            batch_size=32,
            epochs=20,
            workers=-1,
            use_multiprocessing=True,
            callbacks=[
                EarlyStopByAccuracy(0.998),
                TensorBoard(log_dir=self._logs)
            ]
        )
        return history.history
    
    def save_weights(self, _model_weights: str = None) -> None:
        """
        保存权重\n
        :param _model_weights: 权重保存路径
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)

    def load_weights(self, _model_weights: str = None) -> None:
        """
        加载权重\n
        :param _model_weights: 模型权重保存路径
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.load_weights(_model_weights)

    def get_word_index(self, reverse: bool = False) -> dict:
        """
        获取词索引\n
        :param reverse: 是否翻转
        :return: 词索引
        """
        word_index = self._token.word_index
        if not reverse:
            return word_index
        return dict([(value, key) for key, value in word_index.items()])

    def get_weight(self, name: str = None, index: int = None) -> np.ndarray:
        """
        获取某一层的权重\n
        :param name: 层的名字
        :param index: 第几层
        :return: 权重矩阵
        """
        return self._model.get_layer(name, index).get_weights()[0]