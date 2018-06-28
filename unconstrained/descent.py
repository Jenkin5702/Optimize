# coding=utf-8
from numpy import *

from Function import *
from linear_search.gold_division import *


def descent(f, start):
    fun = Function(f)
    x = array(start)
    d = -fun.grad(x)
    i=0
    while fun.norm(x) > 0.01 and i<30:
        alpha = gold_division(f, x, d,[0,12])
        x = x + alpha * array(d)
        d = -fun.grad(x)
        i+=1
    return x