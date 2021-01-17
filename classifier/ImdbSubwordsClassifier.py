#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, History
import tensorflow as tf
import numpy as np
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy


class ImdbSubwordsClassifier:
    _model: Sequential
    _model_weights: str
    _logs: str
    _token: Tokenizer
    num_words: int
    MAX_LENGTH: int = 120

    def __init__(self, _model_weights: str, _logs: str, _num_words: int = 10000) -> None:
        """
        构造方法\n
        :param _model_weights: 模型权重保存路径
        :param _logs: 日志保存路径
        :param _num_words: 最大词数
        """
        super().__init__()
        self._model_weights = _model_weights
        self._logs = _logs
        self.num_words = _num_words
        self._token = Tokenizer(
            num_words=self.num_words,
            oov_token="<OOV>"
        )

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 编译好的模型
        """
        self._model = Sequential([
            Embedding(input_dim=self.num_words, output_dim=16, input_length=self.MAX_LENGTH, name="embedding_1"),
            Bidirectional(LSTM(units=16, return_sequences=False), name="LSTM_2"),
            Dense(units=64, activation="relu", name="dense_3"),
            Dense(units=1, activation="sigmoid", name="dense_4")
        ])
        self._model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return self._model

    def pre_process(self, _x: [str], _y: [int]) -> (np.ndarray, ):
        """
        数据预处理\n
        :param _x: 训练集x
        :param _y: 训练集y
        :return: 预处理后的参数
        """
        return pad_sequences(
            sequences=self._token.texts_to_sequences(_x),
            maxlen=self.MAX_LENGTH,
            padding="pre",
            truncating="post"
        ), np.array(_y)

    @staticmethod
    def pre_process_dataset(_data: tf.raw_ops.PrefetchDataset) -> tf.raw_ops.PaddedBatchDataset:
        """
        数据预处理\n
        :param _data: 如果x和y在一起
        :return: 处理后的结果
        """
        result: tf.raw_ops.ShuffleDataset = _data.shuffle(10000)
        result: tf.raw_ops.PaddedBatchDataset = result.padded_batch(64, tf.compat.v1.data.get_output_shapes(result))
        return result

    def fit(self, _x_train: [str], _y_train: [int], _x_val: [str], _y_val: [int], _data_train: tf.raw_ops.PaddedBatchDataset = None, _data_validation: tf.raw_ops.PaddedBatchDataset = None) -> dict:
        """
        训练模型\n
        :param _x_train: 训练集x
        :param _y_train: 训练集y
        :param _x_val: 验证集x
        :param _y_val: 验证集y
        :param _data_train: 训练集
        :param _data_validation: 验证集
        :return: 训练历史
        """
        if _data_train is None and _data_validation is None:
            self._token.fit_on_texts(_x_train)
            x_train, y_train = self.pre_process(_x_train, _y_train)
            history: History = self._model.fit(
                x=x_train,
                y=y_train,
                validation_data=self.pre_process(_x_val, _y_val),
                epochs=30,
                callbacks=[
                    EarlyStopByAccuracy(0.998),
                    TensorBoard(self._logs)
                ],
                workers=-1,
                use_multiprocessing=True
            )
            return history.history
        else:
            history: History = self._model.fit(
                self.pre_process_dataset(_data_train),
                validation_data=self.pre_process_dataset(_data_validation),
                epochs=30,
                callbacks=[
                    EarlyStopByAccuracy(0.998),
                    TensorBoard(self._logs)
                ],
                workers=-1,
                use_multiprocessing=True
            )
            return history.history

    def save_weights(self, _model_weights: str = None) -> None:
        """
        保存模型权重\n
        :param _model_weights: 模型权重保存路径
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)

    def load_weights(self, _model_weights: str = None) -> None:
        """
        加载模型权重\n
        :param _model_weights: 模型权重保存路径
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)

