# coding=utf-8
from numpy import *

from Function import *
from linear_search.wolfe import *


# 拟牛顿法
def simu_newton(f, start):
    n=size(start)
    fun = Function(f)
    x = array(start)
    g = fun.grad(x)
    B=eye(n)
    while fun.norm(x) > 0.01:
        d = (-dot(linalg.inv(B), g)).tolist()
        alpha = wolfe(f, x, d)
        x_d=array([alpha * array(d)])
        x = x + alpha * array(d)
        g_d=array([fun.grad(x)-g])
        g = fun.grad(x)
        B_d=dot(B,x_d.T)
        B=B+dot(g_d.T,g_d)/dot(g_d,x_d.T)-dot(B_d,B_d.T)/dot(x_d,B_d)
    return x