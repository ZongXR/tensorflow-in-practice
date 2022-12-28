# -*- coding: utf-8 -*-
"""
完全不使用keras高阶API
"""
from typing import Tuple, List
import tensorflow as tf
from tensorflow import Module, Tensor, Variable, GradientTape, random_normal_initializer
from tensorflow_datasets import load
from tensorflow.python.data.ops.dataset_ops import MapDataset, BatchDataset, CacheDataset, PrefetchDataset, ShuffleDataset


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.reset_default_graph()


def gradient_descent(_learning_rate: float, _grads: List[Tensor], _vars: List[Variable]) -> None:
    """
    梯度下降\n
    :param _learning_rate: 学习率
    :param _grads: 梯度
    :param _vars: 参数
    :return: 空
    """
    for i, _x in enumerate(_vars):
        _x.assign(_x - _learning_rate * _grads[i].numpy())


def normalize_img(_image: Tensor, _label: Tensor) -> Tuple[Tensor, Tensor]:
    """
    图像标准化\n
    :param _image: 图像
    :param _label: 标签
    :return: x, y
    """
    return tf.cast(_image, tf.float64) / 255., _label


class ConvNet(Module):

    def __init__(self, batch_size: int, num_classes: int, name=None):
        super(ConvNet, self).__init__(name=name)
        self.batch_size = batch_size
        with self.name_scope:  # 相当于with tf.name_scope("demo_module")
            self.filters1 = Variable(tf.random.normal([3, 3, 1, 64], dtype=tf.float64), trainable=True)
            self.b1 = Variable(random_normal_initializer(mean=0.0, stddev=1.0)(shape=(1, 1, 1, 64), dtype=tf.float64), trainable=True)
            self.filters2 = Variable(tf.random.normal([3, 3, 64, 32], dtype=tf.float64), trainable=True)
            self.b2 = Variable(random_normal_initializer(mean=0.0, stddev=1.0)(shape=(1, 1, 1, 32), dtype=tf.float64), trainable=True)
            self.filters3 = Variable(tf.random.normal([3, 3, 32, 8], dtype=tf.float64), trainable=True)
            self.b3 = Variable(random_normal_initializer(mean=0.0, stddev=1.0)(shape=(1, 1, 1, 8), dtype=tf.float64), trainable=True)
            self.w = Variable(random_normal_initializer(mean=0.0, stddev=1.0)(shape=(3872, num_classes), dtype=tf.float64), trainable=True)
            self.b = Variable(random_normal_initializer(mean=0.0, stddev=1.0)(shape=(batch_size, num_classes), dtype=tf.float64), trainable=True)

    @tf.function
    def __call__(self, x: Tensor) -> Tensor:
        with self.name_scope:
            y = tf.nn.relu(tf.nn.conv2d(x, self.filters1, strides=1, padding="VALID") + self.b1)
            y = tf.nn.relu(tf.nn.conv2d(y, self.filters2, strides=1, padding="VALID") + self.b2)
            y = tf.nn.relu(tf.nn.conv2d(y, self.filters3, strides=1, padding="VALID") + self.b3)
            y = tf.reshape(y, shape=(self.batch_size, -1))
            y = tf.nn.softmax(y @ self.w + self.b)
            return y


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

    model = ConvNet(batch_size=32, num_classes=num_classes)

    for epoch in range(40):
        for x_train, y_train in ds_train:
            with GradientTape() as tape:
                y_pred = model(x_train)
                loss = tf.losses.SparseCategoricalCrossentropy()(y_train, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            gradient_descent(1e-4, grads, model.trainable_variables)
        tf.print("--------------epoch: %d, loss: %f --------------" % (epoch, loss.numpy()))
