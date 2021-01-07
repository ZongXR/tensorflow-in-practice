from classifier.CatDogClassifierInceptionV3 import CatDogClassifierInceptionV3
from tensorflow.keras.backend import clear_session
import tensorflow as tf


if __name__ == '__main__':
    # 设置显存自动增长
    clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 加载模型并训练
    clf = CatDogClassifierInceptionV3(
        _input_shape=(150, 150, 3),
        _logs="./logs/CatDogClassifierInceptionV3",
        _model_weights="./models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    clf.fit("./data/cats_and_dogs_filtered")
    clf.save_weights("./models/CatDogClassifierInceptionV3.h5")