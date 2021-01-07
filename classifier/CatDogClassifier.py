#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.callbacks import TensorBoard
from callbacks.MyCallback import MyCallBack


# %%
class CatDogClassifier:
    _model: Sequential
    _SIZE = (300, 300, 3)
    _model_path: str
    _logs: str

    def __init__(self, _model_path: str, _logs: str):
        """
        构造方法\n
        :param _model_path 模型存储路径
        :param _logs 日志存储路径
        """
        super().__init__()
        self._model_path = _model_path
        self._logs = _logs

    def build_model(self) -> Sequential:
        """
        构建模型\n
        :return: 编译好的模型
        """
        self._model = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_1", input_shape=CatDogClassifier._SIZE),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_1"),
            Conv2D(filters=32, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_2"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_2"),
            Conv2D(filters=64, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_3"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_3"),
            Conv2D(filters=64, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_4"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_4"),
            Conv2D(filters=64, kernel_size=(3, 3), data_format="channels_last", activation="relu", name="conv2d_5"),
            MaxPool2D(pool_size=(2, 2), data_format="channels_last", name="pool2d_5"),
            Flatten(data_format="channels_last", name="flatten_6"),
            Dense(units=512, activation="relu", name="dense_6"),
            Dense(units=1, activation="sigmoid", name="dense_7")
        ])
        self._model.summary()
        self._model.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return self._model

    def fit(self, _data: str) -> None:
        """
        训练模型\n
        :param _data: 数据文件夹
        :return: 空
        """
        try:
            self._model.load_weights(self._model_path)
        except OSError:
            train_data = ImageDataGenerator(rescale=1/255).flow_from_directory(
                directory=_data + "/train",
                target_size=CatDogClassifier._SIZE[:-1],
                class_mode="binary",
                batch_size=32
            )
            validation_data = ImageDataGenerator(rescale=1/255).flow_from_directory(
                directory=_data + "/validation",
                target_size=CatDogClassifier._SIZE[:-1],
                class_mode="binary",
                batch_size=32
            )
            logs = TensorBoard(log_dir=self._logs)
            self._model.fit_generator(
                generator=train_data,
                validation_data=validation_data,
                epochs=20,
                callbacks=[MyCallBack(), logs],
                workers=-1,
                use_multiprocessing=True
            )

    def save_weights(self) -> None:
        """
        保存权重\n
        :return: 空
        """
        self._model.save_weights(self._model_path)


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 加载模型并训练
    clf = CatDogClassifier("../models/CatDogClassifier.h5", "../logs/CatDogClassifier")
    clf.build_model()
    clf.fit("../data/cats_and_dogs_filtered")
    clf.save_weights()
    


