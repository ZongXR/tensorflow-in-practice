#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import matplotlib.pylab as plt
import os
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy


class HorseHumanClassifier:
    _model: Sequential
    _SIZE = (300, 300, 3)
    _model_weights: str
    _logs: str
    _targets: (str, )

    def __init__(self, _model_weights: str, _logs: str) -> None:
        """
        构造方法\n
        :param _model_weights: 模型权重
        :param _logs: 日志
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
            Conv2D(16, (3, 3), activation="relu", name="conv2d_1", input_shape=HorseHumanClassifier._SIZE),
            MaxPool2D(pool_size=(2, 2), name="pool2d_1"),
            Conv2D(32, (3, 3), activation="relu", name="conv2d_2"),
            MaxPool2D(pool_size=(2, 2), name="pool2d_2"),
            Conv2D(64, (3, 3), activation="relu", name="conv2d_3"),
            MaxPool2D(pool_size=(2, 2), name="pool2d_3"),
            Flatten(data_format="channels_last", name="flatten_4"),
            Dense(units=512, activation="relu", name="dense_4"),
            Dense(units=1, activation="sigmoid", name="dense_5")
        ])
        self._model.compile(
            optimizer=RMSprop(lr=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        self._model.summary()
        return self._model

    def fit(self, _data: str) -> Sequential:
        """
        训练新模型\n
        :param _data: 数据文件夹
        :return: 训练好的模型
        """
        self._targets = os.listdir(_data + "/train_data")
        try:
            self._model.load_weights(self._model_weights)
            return self._model
        except OSError:
            train_generator = ImageDataGenerator(rescale=1 / 255).flow_from_directory(
                directory=_data + "/train_data",
                target_size=HorseHumanClassifier._SIZE[:-1],
                class_mode="binary",
                batch_size=128
            )
            validation_generator = ImageDataGenerator(rescale=1 / 255).flow_from_directory(
                directory=_data + "/validation_data",
                target_size=HorseHumanClassifier._SIZE[:-1],
                class_mode="binary",
                batch_size=32
            )
            self._model.fit_generator(
                generator=train_generator,
                validation_data=validation_generator,
                epochs=15,
                # steps_per_epoch=8,
                workers=-1,
                use_multiprocessing=True,
                callbacks=[
                    EarlyStopByAccuracy(0.998),
                    TensorBoard(log_dir=self._logs)
                ]
            )
            return self._model

    def save_weights(self, _model_weights: str) -> None:
        """
        保存权重\n
        :param _model_weights: 权重文件夹
        :return: 空
        """
        if _model_weights is None:
            _model_weights = self._model_weights
        self._model.save_weights(_model_weights)

    def predict(self, _path: str) -> (str, ):
        """
        预测\n
        :param _path: 保存预测样本的文件夹
        :return: 预测的标签
        """
        plt.figure()
        plt.subplots(1, len(os.listdir(_path)))
        assert self._targets is not None, "模型尚未训练"
        _targets = np.array(self._targets, dtype=str)
        _x = []
        for i, _file in enumerate(os.listdir(_path)):
            plt.subplot(1, len(os.listdir(_path)), i + 1)
            _img = img_to_array(load_img(_path + "/" + _file, target_size=HorseHumanClassifier._SIZE[:-1]))
            plt.imshow(_img.astype(int))
            _x.append(_img / 255)
        _predicts = np.reshape(self._model.predict_classes(np.array(_x)), (1, -1))
        plt.tight_layout()
        plt.show()
        return tuple(_targets[_predicts][0])

    def predict_one(self, _file_path: str) -> str:
        """
        预测一个样本\n
        :param _file_path: 文件路径
        :return: 预测的标签
        """
        assert self._targets is not None, "模型尚未训练"
        _targets = np.array(self._targets, dtype=str)
        _img = img_to_array(load_img(_file_path, target_size=HorseHumanClassifier._SIZE[:-1]))
        _x = [_img / 255]
        _predicts = np.reshape(self._model.predict_classes(np.array(_x)), (1, -1))
        return _targets[_predicts][0][0]


