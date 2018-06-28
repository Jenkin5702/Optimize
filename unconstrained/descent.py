# coding=utf-8
from numpy import *

from Function import *
from linear_search.wolfe import *


def descent(f, start):
    fun = Function(f)
    x = array(start)
    d = -fun.grad(x)
    while fun.norm(x) > 0.01:
        alpha = wolfe(f, x, d)
        x = x + alpha * array(d)
        d = -fun.grad(x)
    return x