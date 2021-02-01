from utils.time_series.MixedSeries import MixedSeries
from utils.time_series.TimeSeriesUtils import TimeSeriesUtils
from regressor.TimeSeriesRegressor import TimeSeriesRegressor
from tensorflow.keras.backend import clear_session
from tensorflow.keras.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    clear_session()
    tf.compat.v1.reset_default_graph()
    WINDOW_SIZE = 20
    SPLIT_TIME = 1000
    series = MixedSeries(
        _start=0,
        _end=4 * 365 + 1,
        _slop=0,
        _bias=0,
        _period=365,
        _season_amplitude=40,
        _phase=0,
        _level=5,
        _auto_amplitude=0,
        _seed=42
    )
    x_train = series[0:SPLIT_TIME]
    x_valid = series[SPLIT_TIME:]
    time = np.arange(0, len(series), 1)
    time_train = time[0:SPLIT_TIME]
    time_valid = time[SPLIT_TIME:]
    series.plot()
    data_train: tf.raw_ops.PrefetchDataset = TimeSeriesUtils.split_window(
        series=x_train.get_values(),
        window_size=WINDOW_SIZE,
        batch_size=32,
        buffer_size=1000
    )
    reg = TimeSeriesRegressor(
        _input_length=WINDOW_SIZE,
        _model_weights="./models/TimeSeriesRegressor.h5",
        _logs="./logs/TimeSeriesRegressor"
    )
    reg.build_model()
    reg.fit(data_train)
    y: np.ndarray = reg.predict(TimeSeriesUtils.window(series.get_values(), WINDOW_SIZE, 32))
    y: np.ndarray = y.reshape((y.shape[0], ))
    # 最后一个时间窗口没有真值进行比对，所以截止到此
    y_valid = y[SPLIT_TIME - WINDOW_SIZE:-1]
    print("mae =", mean_absolute_error(x_valid.get_values(), y_valid).numpy())
    plt.figure(1)
    plt.plot(time_valid, x_valid.get_values())
    plt.plot(time_valid, y_valid)
    plt.show()


