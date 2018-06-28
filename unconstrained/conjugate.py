# coding=utf-8
# FR-CG共轭梯度法
import numpy as np

from linear_search.wolfe import *


def conjugate(_f, _x):
    fun = Function(_f)
    x = np.array(_x)
    d = -fun.grad(x)
    while fun.norm(x) > 0.01:
        alpha = wolfe(_f, x, d)
        g = np.mat(fun.grad(x))
        beta = 1 / np.dot(g, g.T)
        x = x + alpha * d
        g = np.mat(fun.grad(x))
        beta = beta * np.dot(g, g.T)
        d = array(-g + beta * d)[0]
    return x