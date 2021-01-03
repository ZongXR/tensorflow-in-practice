import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os


def show_img(_img: np.ndarray) -> None:
    """
    显示图像\n
    :param _img: 样本索引
    :return: 空
    """
    plt.imshow(_img)
    plt.show()


def show_img_info(_imgs: np.ndarray, _targets: np.ndarray, index: int) -> None:
    """
    显示图像信息\n
    :param _imgs: 图像样本集
    :param _targets: 标签
    :param index: 第几个样本
    :return: 空
    """
    show_img(_imgs[index])
    print(_targets[index])


def pre_process(*_datas: np.ndarray) -> (np.ndarray, ):
    """
    数据预处理\n
    :param _datas: 处理前的数据
    :return: 处理后的数据
    """
    result = []
    for _data in _datas:
        result.append(_data / 255.0)
    return tuple(result)


def build_model(input_shape: (int, ), output_shape: int) -> Sequential:
    """
    构建模型\n
    :param input_shape: 输入维度
    :param output_shape: 输出维度
    :return: 编译好的模型
    """
    _model = Sequential([
        Flatten(input_shape=input_shape),
        # Dense前必须接Flatten
        Dense(units=1024, activation=tf.nn.relu),
        Dense(units=output_shape, activation=tf.nn.softmax)
    ])
    _model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return _model


def predict_one(_model: Sequential, _x: np.ndarray) -> (np.ndarray, int):
    """
    预测一个样本\n
    :param _model: 模型
    :param _x: 预测的X
    :return: (softmax, 类别)
    """
    _prob = _model.predict(np.array([_x]))
    return _prob, np.argmax(_prob, axis=-1)


class MyCallBack(Callback):
    """
    回调\n
    """
    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        early stop\n
        :param epoch: 轮数
        :param logs: 日志
        :return: 空
        """
        if logs["accuracy"] > 0.9:
            print()
            print("accuracy到达0.9, epoch=%d" % epoch + 1)
            self.model.stop_training = True


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    show_img_info(x_train, y_train, 5000)
    # 数据预处理
    x_train, x_test = pre_process(x_train, x_test)
    # 构建模型
    model = build_model(x_train.shape[1:], len(np.unique(y_train)))
    # 设置回调
    my_callback = MyCallBack()
    board = TensorBoard(
        log_dir="../logs/%s" % os.path.basename(__file__)
    )
    # 训练
    try:
        model.load_weights("../models/FashionMNIST_Dense.h5")
    except OSError:
        model.fit(
            x=x_train,
            y=y_train,
            epochs=10,
            callbacks=[my_callback, board]
        )
        model.save_weights("../models/FashionMNIST.h5", overwrite=True)
    print(model.evaluate(x=x_test, y=y_test))
    # 预测
    idx = 0
    _, category = predict_one(model, x_test[idx])
    print(category)
    show_img_info(x_test, y_test, idx)


