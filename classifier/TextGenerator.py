#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, LSTM, Dropout, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import History, TensorBoard
from tensorflow.keras.optimizers import Adam


class TextGenerator:
    _model: Sequential
    _model_weights: str
    _logs: str
    _max_length: int
    _token: Tokenizer

    def __init__(self, _num_words: int, _model_weights: str, _logs: str) -> None:
        """
        构造方法\n
        :param _num_words: 词的数量
        :param _model_weights: 模型权重保存位置
        :param _logs: 日志保存位置
        """
        super().__init__()
        self._model_weights = _model_weights
        self._logs = _logs
        self._token = Tokenizer(
            num_words=_num_words,
            oov_token="<OOV>"
        )

    def pre_process(self, _text: str, _init_token: bool = False) -> (np.ndarray, ):
        """
        字符串预处理, 第一次调用本方法必须指定_init_token为True \n
        :param _text: 输入文本
        :param _init_token: 是否初始化_token
        :return: (数据, 标签)
        """
        _texts = _text.split("\n")
        _texts = list(filter(lambda x: len(x) > 0, _texts))
        if _init_token:
            self._token.fit_on_texts(_texts)
        _seq = self._token.texts_to_sequences(_texts)
        _seq = self.__expand_seq(_seq)
        self._max_length = max([len(x) for x in _seq])
        _seq = pad_sequences(
            sequences=_seq,
            maxlen=self._max_length,
            padding="pre",
            truncating="post"
        )
        return _seq[:, :-1], to_categorical(
            y=_seq[:, -1],
            num_classes=len(self._token.word_index)
        )

    @staticmethod
    def __expand_seq(_seq: [[int]]) -> [[int]]:
        """
        扩展序列\n
        :param _seq: 序列
        :return: 扩展后的序列
        """
        results = []
        for _se in _seq:
            for i in range(1, len(_se)):
                results.append(_se[0:i + 1])
        return results

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 编译好的模型
        """
        self._model = Sequential([
            Embedding(input_dim=len(self._token.word_index), output_dim=100, input_length=self._max_length - 1, name="embedding_1"),
            Bidirectional(LSTM(units=150), name="LSTM_2"),
            Dense(units=len(self._token.word_index), activation="softmax", name="dense_3"),
        ])
        self._model.compile(
            optimizer=Adam(lr=0.01),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return self._model

    def fit(self, _x_train: np.ndarray, _y_train: np.ndarray) -> dict:
        """
        训练模型\n
        :param _x_train: 训练集x
        :param _y_train: 训练集y
        :return:训练历史
        """
        history: History = self._model.fit(
            x=_x_train,
            y=_y_train,
            epochs=100,
            workers=-1,
            use_multiprocessing=True,
            callbacks=[
                TensorBoard(self._logs)
            ]
        )
        return history.history

    def generate(self, _text: str, _length: int) -> str:
        """
        生成文本\n
        :param _text: 初始文本
        :param _length: 生成的长度
        :return: 生成文本
        """
        reverse_word_index = dict([(value, key) for key, value in self._token.word_index.items()])
        _input = self._token.texts_to_sequences([_text])
        _input = pad_sequences(
            sequences=_input,
            maxlen=self._max_length,
            padding="pre",
            truncating="pre"
        )
        for _ in range(_length):
            _output: np.ndarray = self._model.predict_classes(np.array(_input))
            if _output[0] in reverse_word_index.keys():
                _text = _text + " " + reverse_word_index[_output[0]]
                _input[0, 0:-1] = _input[0, 1:]
                _input[0, -1] = _output[0]
            else:
                _text = _text + " ?"
                _input[0, 0:-1] = _input[0, 1:]
                _input[0, -1] = self._token.word_index[self._token.oov_token]
        return _text

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
        :param _model_weights:权重路径
        :return:空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.load_weights(_model_weights)


if __name__ == '__main__':
    clf = TextGenerator(263, "../models/TextGenerator.h5", "../logs/TextGenerator")





