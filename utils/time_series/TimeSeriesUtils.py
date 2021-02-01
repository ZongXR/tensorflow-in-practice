import numpy as np
import tensorflow as tf


class TimeSeriesUtils:

    @staticmethod
    def split_window(series: np.ndarray, window_size: int, batch_size: int, buffer_size: int) -> tf.raw_ops.PrefetchDataset:
        """
        划分滑动窗口\n
        :param series: 时间序列
        :param window_size: 窗口大小
        :param batch_size: 批大小
        :param buffer_size:
        :return:
        """
        dataset: tf.raw_ops.TensorSliceDataset = tf.data.Dataset.from_tensor_slices(series)
        dataset: tf.raw_ops.WindowDataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
        dataset: tf.raw_ops.FlatMapDataset = dataset.flat_map(lambda x: x.batch(window_size + 1))
        dataset: tf.raw_ops.ShuffleDataset = dataset.shuffle(buffer_size=buffer_size)
        dataset: tf.raw_ops.MapDataset = dataset.map(lambda x: (x[0:-1], x[-1]))
        dataset: tf.raw_ops.BatchDataset = dataset.batch(batch_size=batch_size)
        dataset: tf.raw_ops.PrefetchDataset = dataset.prefetch(1)
        return dataset

    @staticmethod
    def window(series: np.ndarray, window_size: int, batch_size: int) -> tf.raw_ops.PrefetchDataset:
        """
        划分窗口\n
        :param series: 时间序列
        :param window_size: 窗口长度
        :param batch_size: 批大小
        :return: 最终数据
        """
        dataset: tf.raw_ops.TensorSliceDataset = tf.data.Dataset.from_tensor_slices(series)
        dataset: tf.raw_ops.WindowDataset = dataset.window(size=window_size, shift=1, drop_remainder=True)
        dataset: tf.raw_ops.FlatMapDataset = dataset.flat_map(lambda x: x.batch(window_size))
        dataset: tf.raw_ops.BatchDataset = dataset.batch(batch_size=batch_size)
        dataset: tf.raw_ops.PrefetchDataset = dataset.prefetch(1)
        return dataset
