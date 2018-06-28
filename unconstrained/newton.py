# coding=utf-8
from linear_search.gold_division import *
from linear_search.wolfe import *


# (带步长因子的)牛顿法
def newton(f, start):
    fun = Function(f)
    x = array(start)
    g = fun.grad(x)
    while fun.norm(x) > 0.01:
        G = fun.hesse(x)
        d = (-dot(linalg.inv(G), g)).tolist()[0]
        alpha = wolfe(f, x, d)
        x = x + alpha * array(d)
        g = fun.grad(x)
    return x


def f(val):
    return 100*(val[0]**2-val[1])**2+(val[0]-1)**2

print(newton(f,[1,2]))