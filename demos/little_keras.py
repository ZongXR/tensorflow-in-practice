# -*- coding: utf-8 -*-
"""
使用一部分keras高阶API
"""
from typing import Tuple
import tensorflow as tf
from tensorflow import GradientTape, Tensor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import clear_session
from tensorflow_datasets import load
from tensorflow.python.data.ops.dataset_ops import MapDataset, BatchDataset, CacheDataset, PrefetchDataset, ShuffleDataset


clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def normalize_img(_image: Tensor, _label: Tensor) -> Tuple[Tensor, Tensor]:
    """
    图像标准化\n
    :param _image: 图像
    :param _label: 标签
    :return: x, y
    """
    return tf.cast(_image, tf.float64) / 255., _label


if __name__ == '__main__':
    (ds_train, ds_test), ds_info = load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train: MapDataset = ds_train.map(normalize_img, num_parallel_calls=8)
    ds_train: CacheDataset = ds_train.cache()
    ds_train: ShuffleDataset = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train: BatchDataset = ds_train.batch(32)
    ds_train: PrefetchDataset = ds_train.prefetch(8)
    num_classes = ds_info.features["label"].num_classes
    h, w, c = ds_info.features.shape["image"]

    model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(h, w, c)),
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
        Flatten(data_format="channels_last"),
        Dense(units=num_classes, activation="softmax")
    ])
    optimizer = SGD(learning_rate=1e-4)

    for epoch in range(40):
        for x_train, y_train in ds_train:
            with GradientTape() as tape:
                y_pred = model(x_train)
                loss = SparseCategoricalCrossentropy()(y_train, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        tf.print("--------------epoch: %d, loss: %f --------------" % (epoch, loss.numpy()))

