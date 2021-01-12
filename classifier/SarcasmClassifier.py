#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy


class SarcasmClassifier:
    _model: Sequential
    _model_weights: str
    _logs: str
    _token: Tokenizer
    _num_words: int
    _max_length: int

    def __init__(self, _model_weights: str, _logs: str) -> None:
        """
        构造方法\n
        :param _model_weights: 模型权重存储位置\n
        :param _logs: 日志存储位置
        """
        self._model_weights = _model_weights
        self._logs = _logs
        self._num_words = 10000
        self._max_length = 120
        self._token = Tokenizer(
            num_words=self._num_words,
            oov_token="<OOV>"
        )

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 构建好的模型
        """
        self._model = Sequential([
            Embedding(input_dim=self._num_words, output_dim=16, input_length=self._max_length, name="embedding_1"),
            GlobalAveragePooling1D(name="pool1d_2"),
            Dense(units=24, activation="relu", name="dense_2"),
            Dense(units=1, activation="sigmoid", name="dense_3")
        ])
        self._model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return self._model

    def pre_process(self, _x: [str], _y: [int]) -> (np.ndarray, ):
        """
        数据预处理\n
        :param _x: 数据
        :param _y: 标签
        :return: 处理之后的结果
        """
        return pad_sequences(
            sequences=self._token.texts_to_sequences(_x),
            maxlen=self._max_length,
            padding="post",
            truncating="post"
        ), np.array(_y)

    def fit(self, _x_train: [str], _y_train: [int], _x_validation: [str], _y_validation: [int]) -> dict:
        """
        训练模型\n
        :param _x_train: 训练集x
        :param _y_train: 训练集y
        :param _x_validation: 验证集x
        :param _y_validation: 验证集y
        :return: 训练历史过程
        """
        self._token.fit_on_texts(_x_train)
        _xx_train, _yy_train = self.pre_process(_x_train, _y_train)
        history = self._model.fit(
            x=_xx_train,
            y=_yy_train,
            validation_data=self.pre_process(_x_validation, _y_validation),
            batch_size=32,
            epochs=30,
            callbacks=[
                EarlyStopByAccuracy(0.998),
                TensorBoard(self._logs)
            ]
        )
        return history.history

    def save_weights(self, _model_weights: str = None) -> None:
        """
        保存权重\n
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
        self._model.load_weights(_model_weights)

    def words_vectors(self, words_file: str, vectors_file: str) -> None:
        """
        保存词向量
        :param words_file: 词文件保存路径
        :param vectors_file: 向量文件保存路径
        :return: 空
        """
        if not words_file.endswith(".tsv"):
            words_file = words_file + ".tsv"
        if not vectors_file.endswith(".tsv"):
            vectors_file = vectors_file + ".tsv"
        layer = self._model.get_layer(name="embedding_1")
        weight = layer.get_weights()[0]
        word_index = self._token.word_index
        reverse_word_index = dict([(value, key) for key, value in word_index.items()])
        f_words = open(words_file, "w+", encoding="utf8")
        f_vec = open(vectors_file, "w+", encoding="utf8")
        for index, word in reverse_word_index.items():
            if index - 1 < weight.shape[0]:
                f_words.write(word + "\n")
                f_vec.write("\t".join(weight[index - 1, :].astype(str).tolist() + ["\n"]))
        f_vec.close()
        f_words.close()


