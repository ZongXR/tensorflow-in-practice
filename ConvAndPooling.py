import numpy as np
from scipy.misc import ascent
from utils.layers import conv, max_pooling
from utils.images import show_img


if __name__ == '__main__':
    img = ascent()
    filtor = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    img_conv = conv(img, filtor)
    img_pool = max_pooling(img_conv, (2, 2))
    show_img(("./imgs/origin.png", img), ("./imgs/conv.png", img_conv), ("./imgs/max_pool.png", img_pool))