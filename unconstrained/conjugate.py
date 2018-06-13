# coding=utf-8
# FR-CG共轭梯度法
from linear_search.Function import *
from linear_search.wolfe import *
import numpy as np


def conjugate(_f, _x):
    fun = Function(_f)
    x = array(_x)
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