#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy
import numpy as np
from joblib import dump


class RockPaperScissorsClassifier:
    _model: Sequential
    _input_shape: (int, )
    _model_weights: str
    _model_path: str
    _logs: str
    _targets: (str, )

    def __init__(self, _model_weights: str, _logs: str, _model_path: str = None, _input_shape: (int, ) = (300, 300, 3)) -> None:
        """
        构造方法\n
        :param _input_shape: 输入大小
        :param _model_weights: 权重路径
        :param _model_path: 模型路径
        :param _logs: 日志路径
        """
        super().__init__()
        self._model_weights = _model_weights
        self._logs = _logs
        self._input_shape = _input_shape
        if _model_path is not None:
            try:
                self._model = load_model(_model_path)
            except OSError:
                print("未找到模型")

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 编译好的模型
        """
        self._model = Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_1", input_shape=self._input_shape),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_1"),
            Conv2D(filters=64, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_2"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_2"),
            Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_3"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_3"),
            Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_4"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_4"),
            Flatten(data_format="channels_last", name="Flatten_5"),
            Dropout(rate=0.2, name="dropout_5"),
            Dense(units=512, activation="relu", name="dense_5"),
            Dense(units=3, activation="sigmoid", name="dense_6")
        ])
        self._model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self._model.summary()
        return self._model

    def fit(self, _train: str, _validation: str) -> dict:
        """
        训练模型\n
        :param _train: 保存训练集x的文件夹
        :param _validation: 保存训练集y的文件夹
        :return: 训练历史数据
        """
        _train_data = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode="nearest",
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1/255.0
        ).flow_from_directory(
            directory=_train,
            target_size=self._input_shape[:-1],
            class_mode="categorical",
            batch_size=32
        )
        self._targets = tuple(_train_data.class_indices.keys())
        _validation_data = ImageDataGenerator(
            rescale=1/255.0
        ).flow().flow_from_directory(
            directory=_validation,
            target_size=self._input_shape[:-1],
            class_mode="categorical",
            batch_size=32
        )
        _history = self._model.fit_generator(
            generator=_train_data,
            validation_data=_validation_data,
            epochs=20,
            workers=-1,
            use_multiprocessing=True,
            callbacks=[
                EarlyStopByAccuracy(0.998),
                TensorBoard(log_dir=self._logs)
            ]
        )
        return _history.history

    def save(self, _model_path: str = None) -> None:
        """
        保存模型\n
        :param _model_path: 模型路径
        :return: 空
        """
        if _model_path is None:
            _model_path = self._model_path
        self._model.save(_model_path)

    def save_weights(self, _model_weights: str = None) -> None:
        """
        保存权重\n
        :param _model_weights: 模型权重路径
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)

    def predict_one(self, _filepath: str) -> str:
        """
        进行分类预测\n
        :param _filepath: 预测样本路径
        :return: 预测标签
        """
        _img = img_to_array(load_img(_filepath, target_size=self._input_shape[:-1]))
        _x = [_img / 255]
        _predicts = np.reshape(self._model.predict_classes(np.array(_x)), (1, -1))
        return self._targets[int(np.argmax(self._model.predict(np.array(_x))))]

    def save_all(self, _path: str) -> None:
        """
        保存分类器\n
        :param _path: 保存路径
        :return: 空
        """
        dump(self, _path)
