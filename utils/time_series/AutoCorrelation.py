#!/usr/bin/env python
# coding: utf-8

from utils.time_series.TimeSeries import TimeSeries
import numpy as np


class AutoCorrelation(TimeSeries):
    _start: int
    _end: int

    def __init__(self, _start: int, _end: int, _amplitude: int, _seed: int = None) -> None:
        """
        自相关序列\n
        :param _start: 开始索引
        :param _end: 结束索引
        :param _amplitude: 幅度
        :param _seed: 种子
        """
        _x = np.arange(_start, _end)
        super().__init__("autocorrelation", self.__auto_correlation_1(_x, _amplitude, _seed), _start, _end)

    @staticmethod
    def __auto_correlation_1(_time: np.ndarray, _amplitude: int, seed: int = None) -> np.ndarray:
        """
        自相关函数1\n
        :param _time: 时间序列
        :param _amplitude: 幅度
        :param seed: 种子
        :return: 返回值
        """
        rnd = np.random.RandomState(seed)
        fi1 = 0.5
        fi2 = -0.1
        ar = rnd.randn(len(_time) + 50)
        ar[:50] = 100
        for step in range(50, len(_time) + 50):
            ar[step] += fi1 * ar[step - 50]
            ar[step] += fi2 * ar[step - 33]
        return ar[50:] * _amplitude

    @staticmethod
    def __auto_correlation_2(_time: np.ndarray, _amplitude: int, seed: int = None) -> np.ndarray:
        """
        自相关序列\n
        :param _time: 时间序列
        :param _amplitude: 幅度
        :param seed: 种子
        :return: 返回值
        """
        rnd = np.random.RandomState(seed)
        fi = 0.8
        ar = rnd.randn(len(_time) + 1)
        for step in range(1, len(_time) + 1):
            ar[step] += fi * ar[step - 1]
        return ar[1:] * _amplitude


if __name__ == '__main__':
    auto = AutoCorrelation(0, 1463, 10, None)
    auto.plot()