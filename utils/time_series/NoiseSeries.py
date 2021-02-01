#!/usr/bin/env python
# coding: utf-8

import numpy as np
from utils.time_series.TimeSeries import TimeSeries


class NoiseSeries(TimeSeries):
    _start: int
    _end: int

    def __init__(self, _start: int, _end: int, _level: int, _seed: int) -> None:
        """
        噪声序列\n
        :param _level: 噪声等级
        :param _seed: 种子
        :param _start: 开始索引
        :param _end: 结束索引
        """
        super().__init__("noise", np.random.RandomState(_seed).randn(_end - _start) * _level, _start, _end)


if __name__ == '__main__':
    curve = NoiseSeries(0, 100, 1, 42)
    curve.plot()