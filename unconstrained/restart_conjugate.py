# coding=utf-8
# 再开始共轭梯度法
import numpy as np

from linear_search.wolfe import *


def restart_conjugate(_f, _x, n):
    fun = Function(_f)
    x = array(_x)
    while True:
        d = -fun.grad(x)
        k=0
        if(np.linalg.norm(d)<0.01):
            break
        while fun.norm(x) > 0.01:
            g = np.mat(fun.grad(x))

            alpha = wolfe(_f, x, d)
            x = x + alpha * d
            k=k+1

            g1= np.mat(fun.grad(x))

            if np.dot(g1,g.T)/np.dot(g1,g1.T)>0.1 or k>=n:
                if np.linalg.norm(g1)<0.01:
                    return x
                break
            else:
                beta = np.dot(g1, g1.T) / np.dot(g, g.T)
                d = array(-g + beta * d)[0]
                if np.dot(mat(d),g1.T)>0:
                    break
    return x