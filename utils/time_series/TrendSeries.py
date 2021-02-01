#!/usr/bin/env python
# coding: utf-8

from utils.time_series.TimeSeries import TimeSeries
import numpy as np


class TrendSeries(TimeSeries):
    _start: int
    _end: int

    def __init__(self, _start: int, _end: int, _slop: float, _bias: float) -> None:
        """
        构造方法\n
        :param _slop: 倾斜度
        :param _bias: 偏置
        :param _start: 开始索引
        :param _end: 结束索引
        """
        super().__init__("trend", _slop * np.arange(_start, _end) + _bias, _start, _end)

    def plot(self, _start: int = None, _end: int = None, _format: str = "-", _label: str = None, _file_path: str = None) -> None:
        """
        画图\n
        :param _start: 开始索引
        :param _end: 结束索引
        :param _format: 线形
        :param _label: 标签
        :param _file_path: 文件保存路径
        :return: 空
        """
        if _start is None and _end is None:
            super().plot(self._start, self._end, _format, _label, _file_path)
        else:
            super().plot(_start, _end, _format, _label, _file_path)


if __name__ == '__main__':
    curve = TrendSeries(0, 10, 0.1, 2)
    curve.plot()