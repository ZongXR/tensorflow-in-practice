import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
import os


def show_imgs(idxs: (int, ), path: str) -> None:
    """
    显示样本图片\n
    :param idxs: 样本索引
    :param path: 样本路径
    :return: 空
    """
    plt.figure(1)
    plt.subplots(len(os.listdir(path)), len(idxs))
    for i in range(0, len(os.listdir(path))):
        for j, jj in enumerate(idxs):
            idx = j + 1 + i * len(idxs)
            plt.subplot(len(os.listdir(path)), len(idxs), idx)
            _img = mpimg.imread("%s/%s/%s" % (path, os.listdir(path)[i], os.listdir(path + "/" + os.listdir(path)[i])[jj]))
            plt.imshow(_img)
            plt.title(os.listdir(path + "/" + os.listdir(path)[i])[jj])
    plt.tight_layout()
    plt.savefig("../imgs/人马图片.png")
    plt.show()


def build_model(input_shape: (int, ), output_shape: int) -> Sequential:
    """
    构建模型并编译\n
    :param input_shape: 输入大小
    :param output_shape: 输出大小
    :return: 编译好的模型
    """
    _model = Sequential([
        Conv2D(16, (3, 3), activation="relu", input_shape=input_shape, name="conv2d_1"),
        MaxPool2D(pool_size=(2, 2), name="pool2d_1"),
        Conv2D(32, (3, 3), activation="relu", name="conv2d_2"),
        MaxPool2D(pool_size=(2, 2), name="pool2d_2"),
        Conv2D(64, (3, 3), activation="relu", name="conv2d_3"),
        MaxPool2D(pool_size=(2, 2), name="pool2d_3"),
        Flatten(data_format="channels_last", name="flatten_4"),
        Dense(units=512, activation="relu", name="dense_4"),
        Dense(units=output_shape, activation="sigmoid", name="dense_5")
    ])
    _model.compile(
        optimizer=RMSprop(lr=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    _model.summary()
    return _model


def predict(_model: Sequential, _path: str) -> (str, ):
    """
    预测一系列样本\n
    :param _model: 训练好的模型
    :param _path: 预测样本
    :return: 预测标签
    """
    plt.figure(2)
    plt.subplots(1, len(os.listdir(_path)))
    _targets = np.array(["马", "人"], dtype=str)
    _x = []
    for i, _file in enumerate(os.listdir(_path)):
        plt.subplot(1, len(os.listdir(_path)), i + 1)
        _img = img_to_array(load_img(_path + "/" + _file, target_size=(300, 300)))
        plt.imshow(_img.astype(int))
        _x.append(_img / 255)
    _predicts = np.reshape(_model.predict_classes(np.array(_x)), (1, -1))
    plt.tight_layout()
    plt.savefig("../imgs/人马预测.png")
    plt.show()
    print(_targets[_predicts][0])


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
        if logs["accuracy"] > 0.998:
            print("\n精度已达到0.998，停止训练")
            self.model.stop_training = True


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 显示图片
    print(os.listdir("../data/train_data/horses")[0:10])
    print(os.listdir("../data/train_data/humans")[0:10])
    print(len(os.listdir("../data/train_data/horses")))
    print(len(os.listdir("../data/train_data/humans")))
    show_imgs((1, 20, 300, 43), "../data/train_data")
    train_generator = ImageDataGenerator(rescale=1/255).flow_from_directory(
        directory="../data",
        target_size=(300, 300),
        class_mode="binary",
        batch_size=128
    )
    model = build_model((300, 300, 3), 1)
    try:
        model.load_weights("../models/HorseAndHuman.h5")
    except OSError:
        model.fit_generator(
            generator=train_generator,
            epochs=15,
            # steps_per_epoch=8,
            callbacks=[MyCallback()]
        )
        model.save_weights("../models/HorseAndHuman.h5")
    predict(model, "../data/predict")




