#!/usr/bin/env python
# coding: utf-8
import numpy as np

from utils.time_series.NoiseSeries import NoiseSeries
from utils.time_series.TrendSeries import TrendSeries
from utils.time_series.AutoCorrelation import AutoCorrelation
from utils.time_series.SeasonSeries import SeasonSeries
from utils.time_series.TimeSeries import TimeSeries


class MixedSeries(TimeSeries):
    _start: int
    _end: int

    def __init__(self, _start: int, _end: int, _slop: float, _bias: float, _period: int, _season_amplitude: int, _phase: int, _level: int, _auto_amplitude: int, _seed: int = None) -> None:
        """
        构造方法\n
        :param _start: 开始索引
        :param _end: 结束索引
        :param _slop: 倾斜度
        :param _bias: 偏置
        :param _period: 周期
        :param _season_amplitude: 周期幅度
        :param _phase: 相位
        :param _level: 噪声等级
        :param _auto_amplitude: 自相关幅度
        :param _seed: 种子
        """
        trend = TrendSeries(_start, _end, _slop, _bias)
        season = SeasonSeries(_start, _end, _period, _season_amplitude, _phase)
        noise = NoiseSeries(_start, _end, _level, _seed)
        auto = AutoCorrelation(_start, _end, _auto_amplitude, _seed)
        super().__init__("mixed", trend.get_values() + season.get_values() + noise.get_values() + auto.get_values(), _start, _end)
        

if __name__ == '__main__':
    mixed = MixedSeries(0, 1436, 0.2, 0.1, 365, 10, 0, 1, 2, 42)
    mixed.plot()



