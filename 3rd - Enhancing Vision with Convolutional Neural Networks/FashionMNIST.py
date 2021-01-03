import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import Callback, TensorBoard
import os
import matplotlib.pyplot as plt


def build_model(input_shape: (int, ), output_shape: int) -> Sequential:
    """
    构建模型\n
    :param input_shape: 一个样本的shape
    :param output_shape: 输出的维度
    :return: 编译好的模型
    """
    _model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation="relu", name="conv2d_1"),
        MaxPool2D(pool_size=(2, 2), name="pool2d_1"),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
        MaxPool2D(pool_size=(2, 2), name="pool2d_2"),
        Flatten(name="flatten_3"),
        Dense(units=128, activation="relu", name="dense_3"),
        Dense(units=output_shape, activation="softmax", name="softmax_4")
    ])
    _model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    _model.summary()
    return _model


def pre_process(*_datas: np.ndarray) -> (np.ndarray, ):
    """
    数据预处理
    :param _datas: 处理前的数据
    :return: 处理后的数据
    """
    result = []
    for _data in _datas:
        _data = np.expand_dims(_data, 3)
        result.append(_data / 255)
    return tuple(result)


class MyCallback(Callback):
    """
    自定义回调\n
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        early stop\n
        :param epoch: 轮数
        :param logs: 日志
        :return: 空
        """
        if logs["accuracy"] > 0.95:
            print("\n训练集精度已到达0.95，轮数为%d" % (epoch + 1))
            self.model.stop_training = True


# %% 检视内部
def show_inner(_model: Sequential, _x: np.ndarray, idxs: (int, )) -> None:
    """
    显示内部神经元激活\n
    :param _model: 训练好的模型
    :param _x: 输入数据
    :param idxs: 第几个样本
    :return: 空
    """
    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplots(len(idxs), 4)
    channel_index = 1
    outputs = [x.output for x in _model.layers]
    activation_model = Model(inputs=_model.inputs, outputs=outputs)
    for i in range(0, 4):
        output = activation_model.predict(_x[idxs, :, :, :])[i]
        for k, img in enumerate(output):
            plt.subplot(len(idxs), 4, k * 4 + i + 1)
            plt.imshow(img[:, :, channel_index], cmap='inferno')
            plt.title("样本%d的%s层" % (k, _model.layers[i].name))
    plt.tight_layout()
    plt.savefig("../imgs/inner_weights.png")
    plt.show()


# %% 运行
if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 提取数据
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = pre_process(x_train, x_test)
    model = build_model(x_train.shape[1:], len(np.unique(x_test)))
    try:
        model.load_weights("../models/FashionMNIST_Conv.h5")
    except OSError:
        model.fit(
            x_train,
            y_train,
            epochs=10,
            callbacks=[MyCallback(), TensorBoard(log_dir="../logs/%s" % os.path.basename(__file__))]
        )
        model.save_weights("../models/FashionMNIST_Conv.h5")
    print(model.evaluate(x_test, y_test))
    show_inner(model, x_test, (0, 7, 26))



