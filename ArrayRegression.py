from regressor.LinearRegressor import LinearRegressor
import tensorflow as tf
import numpy as np
import pprint


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    reg = LinearRegressor("./models/ArrayRegression.h5")
    x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    reg.build_model([1])
    pprint.pprint(reg.fit(_x_train=x, _y_train=y).history)
    reg.save_weights("./models/ArrayRegression.h5")
    print(reg.predict_one([10]))



