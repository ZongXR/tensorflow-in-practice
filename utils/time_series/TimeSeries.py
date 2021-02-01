#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class TimeSeries:
    _values: np.ndarray
    _type: str
    _start: int
    _end: int

    def __init__(self, _type: str, _values: np.ndarray, _start: int = 0, _end: int = None) -> None:
        """
        构造方法\n
        :param _type: 时间序列类型
        :param _values: 数值
        :param _start: 开始索引
        :param _end: 结束索引
        """
        super().__init__()
        self._type = _type
        self._values = _values
        self._start = _start
        self._end = _end

    def plot(self, _start: int = 0, _end: int = None, _format: str = "-", _label: str = None, _file_path: str = None) -> None:
        """
        画出时间序列图\n
        :param _start: 开始索引
        :param _end: 结束索引
        :param _format: 格式
        :param _label: 标签
        :param _file_path: 文件保存路径
        :return: 空
        """
        plt.figure(1)
        _y = self._values[_start:_end]
        _x = np.linspace(_start, len(_y) - 1, len(_y))
        plt.plot(_x, _y, _format, label=_label)
        plt.xlabel("Time")
        plt.ylabel("Values")
        if _label:
            plt.legend(font_size=14)
        plt.tight_layout()
        plt.grid(True)
        if _file_path:
            plt.savefig(_file_path)
        plt.show()

    def __add__(self, other):
        """
        连接序列\n
        :param other: 另一个序列
        :return: 连接后的序列
        """
        result = TimeSeries(
            _type=self._type + ", " + other.get_type(),
            _values=np.array(self._values.tolist() + other.get_values().tolist()),
            _start=self._start,
            _end=other.get_end() - other.get_start() + 1 + self._end
        )

    def get_type(self) -> str:
        """
        获取类型\n
        :return: 类型
        """
        return self._type

    def get_values(self) -> np.ndarray:
        """
        返回数值\n
        :return: 序列
        """
        return self._values

    def get_end(self) -> int:
        """
        获取结束索引\n
        :return: 结束索引
        """
        return self._end

    def get_start(self) -> int:
        """
        获取开始索引\n
        :return: 开始索引
        """
        return self._start

    def __len__(self) -> int:
        """
        获取时间序列长度\n
        :return: 时间序列长度
        """
        return len(self._values)

    def __getitem__(self, n: Union[int, slice]):
        """
        切片\n
        :param n: 切片
        :return:切片
        """
        if isinstance(n, slice):
            return TimeSeries(
                _type=self._type,
                _values=self._values[n.start:n.stop:n.step],
                _start=0,
                _end=None
            )
        else:
            return self._values[n]