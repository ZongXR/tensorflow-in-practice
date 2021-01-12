#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, History
from classifier.CatDogClassifier import CatDogClassifier
from callbacks.EarlyStopByAccuracy import EarlyStopByAccuracy


class CatDogClassifierInceptionV3(CatDogClassifier):
    """
    基于InceptionV3的迁移模型分类器
    """
    _model: Model
    _model_weights: str
    _logs: str
    _input_shape: (int, )

    def __init__(
            self,
            _input_shape: (int, ),
            _logs: str,
            _model_weights: str = "imagenet",
    ) -> None:
        """
        构造方法\n
        :param _input_shape: 输入大小
        :param _logs: 日志文件夹
        :param _model_weights: 保存权重的文件
        """
        super().__init__(_model_weights, _logs)
        self._model_weights = _model_weights
        self._logs = _logs
        self._input_shape = _input_shape
        self.build_model()

    def build_model(self) -> Model:
        pre_model = InceptionV3(
            include_top=False,
            input_shape=self._input_shape,
            weights=self._model_weights
        )
        for layer in pre_model.layers:
            layer.trainable = False
        last_layer = pre_model.get_layer("mixed7")
        _output = last_layer.output
        _output = Flatten(name="my_flatten1")(_output)
        _output = Dense(units=1024, activation="relu", name="my_dense1")(_output)
        _output = Dropout(rate=0.2, name="my_dropout")(_output)
        _output = Dense(units=1, activation="sigmoid", name="my_dense2")(_output)
        self._model = Model(pre_model.input, _output)
        self._model.compile(
            optimizer=RMSprop(lr=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        self._model.summary()
        return self._model

    def fit(self, _data: str) -> History:
        train_data = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
            rescale=1 / 255
        ).flow_from_directory(
            directory=_data + "/train",
            target_size=self._input_shape[:-1],
            class_mode="binary",
            batch_size=32
        )
        validation_data = ImageDataGenerator(rescale=1 / 255).flow_from_directory(
            directory=_data + "/validation",
            target_size=self._input_shape[:-1],
            class_mode="binary",
            batch_size=32
        )
        logs = TensorBoard(log_dir=self._logs)
        _history = self._model.fit_generator(
            generator=train_data,
            validation_data=validation_data,
            epochs=100,
            callbacks=[EarlyStopByAccuracy(0.998), logs],
            workers=-1,
            use_multiprocessing=True
        )
        return _history

    def save_weights(self, _my_weights: str = None) -> None:
        """
        保存权重\n
        :param _my_weights:
        :return:
        """
        super().save_weights(_model_weights=_my_weights)

    def show_inner(self, _filepath: str, _savepath: str) -> None:
        super().show_inner(_filepath, _savepath)





