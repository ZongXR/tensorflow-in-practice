#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import TensorBoard, History
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
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

    def fit(self, _data: str) -> Union[History, None]:
        """
        训练模型\n
        :param _data: 数据文件夹
        :return: 空
        """
        try:
            self._model.load_weights(self._model_path)
            return None
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
            _history = self._model.fit_generator(
                generator=train_data,
                validation_data=validation_data,
                epochs=20,
                callbacks=[MyCallBack(), logs],
                workers=-1,
                use_multiprocessing=True
            )
            return _history

    def save_weights(self) -> None:
        """
        保存权重\n
        :return: 空
        """
        self._model.save_weights(self._model_path)

    def show_inner(self, _filepath: str, _savepath: str) -> None:
        """
        显示内部\n
        :param _filepath: 输入样本
        :param _savepath: 保存路径
        :return: 空
        """
        successive_outputs = [layer.output for layer in self._model.layers[1:]]
        visualization_model = Model(inputs=self._model.input, outputs=successive_outputs)
        _x = img_to_array(load_img(_filepath, target_size=CatDogClassifier._SIZE[:-1]))
        _x = np.expand_dims(_x, 0)
        _x = _x / 255.0
        successive_feature_maps = visualization_model.predict(_x)
        layer_names = [layer.name for layer in self._model.layers]
        for c, (layer_name, feature_map) in enumerate(zip(layer_names, successive_feature_maps)):
            if self.filter_conv_pool(layer_name):
                # Just do this for the conv / maxpool layers, not the fully-connected layers
                n_features = feature_map.shape[-1]  # number of features in the feature map
                size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)
                # We will tile our images in this matrix
                display_grid = np.zeros((size, size * n_features))
                # Postprocess the feature to be visually palatable
                for i in range(n_features):
                    x = feature_map[0, :, :, i]
                    x -= x.mean()
                    x /= x.std()
                    x *= 64
                    x += 128
                    x = np.clip(x, 0, 255).astype('uint8')
                    display_grid[:, i * size: (i + 1) * size] = x  # Tile each filter into a horizontal grid
                # Display the grid
                scale = 20. / n_features
                plt.figure(c+1, figsize=(scale * n_features, scale))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                plt.savefig(_savepath + "/%s.jpg" % layer_name)
                plt.show()

    @staticmethod
    def filter_conv_pool(layer_name: str) -> int:
        """
        筛选出卷积层和池化层\n
        :param layer_name: 层的名称
        :return: 是否是卷积或池化
        """
        if layer_name.startswith("conv"):
            return 1
        if layer_name.startswith("pool"):
            return 2
        return 0


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
    


