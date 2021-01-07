from classifier.FashionClassifier import FashionClassifier
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 提取数据
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = FashionClassifier.pre_process(x_train, x_test)
    clf = FashionClassifier("./models/FashionClassifier.h5", "./logs/FashionClassifier")
    clf.build_model()
    clf.fit(x_train, y_train)
    clf.save_weights()
    clf.evaluate(x_test, y_test)
    clf.show_inner(x_test, (0, 7, 26), "./imgs/inner_weights.png")
