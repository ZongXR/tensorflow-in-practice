#!/usr/bin/env python
# coding: utf-8

import numpy as np
from utils.time_series.TimeSeries import TimeSeries


class SeasonSeries(TimeSeries):
    _start: int
    _end: int

    def __init__(self, _start: int, _end: int, _period: int, _amplitude: int, _phase: int) -> None:
        """
        构造方法\n
        :param _period: 周期
        :param _amplitude: 振幅
        :param _phase: 相位
        :param _start: 开始索引
        :param _end: 结束索引
        """
        _season_time = ((np.arange(_start, _end) + _phase) % _period) / _period
        super().__init__("season", _amplitude * self.__seasonal_pattern(_season_time), _start, _end)

    @staticmethod
    def __seasonal_pattern(_time: np.ndarray) -> np.ndarray:
        """
        周期模式\n
        :param _time: 时间
        :return: 序列
        """
        return np.where(
            _time < 0.4,
            np.cos(_time * 2 * np.pi),
            1 / np.exp(3 * _time)
        )

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
    curve = SeasonSeries(0, 1462, 365, 40, 0)
    curve.plot()


