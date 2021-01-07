#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np


def show_img(*_imgs: (str, np.ndarray)) -> None:
    """
    显示图片\n
    :param _imgs: 要显示的图片
    :return: 空
    """
    for i, (_filename, _img) in enumerate(_imgs):
        plt.figure(i)
        plt.grid(False)
        plt.gray()
        plt.axis("off")
        plt.imshow(_img)
        plt.tight_layout()
        plt.savefig(_filename)
        plt.show()
