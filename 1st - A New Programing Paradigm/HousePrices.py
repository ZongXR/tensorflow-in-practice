#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from IPython import get_ipython


def house_model(y_new):
    """
    构建模型，训练并预测\n
    :param y_new: 预测样本
    :return: 预测值
    """
    xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    ys = np.array([1, 1.5, 2, 2.5, 3, 3.5], dtype=float)
    model = Sequential([Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(x=xs, y=ys, epochs=500)
    return model.predict(y_new)[0]


if __name__ == '__main__':
    # 设置显存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, enable=True)
    prediction = house_model([7.0])
    print(prediction)
    get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

