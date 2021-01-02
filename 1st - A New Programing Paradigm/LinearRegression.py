import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    # 构建模型并训练
    model = Sequential([Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    model.fit(x=x, y=y, epochs=500)
    print("f(10) = %f" % model.predict([10]))

