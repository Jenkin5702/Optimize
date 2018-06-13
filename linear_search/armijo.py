from linear_search.Function import *
from numpy import *


def armijo(_f, val, d):
    alpha=1

    def _fi(_alpha):
        return _f(val + _alpha * array(d))

    fi = Function(_fi)
    f = Function(_f)
    fi_=fi.value(alpha)
    f_=f.value(val)
    while True:
        if fi_>f_+1/3*alpha*dot(f.grad(val),array(d).T):
            alpha=0.7*alpha
        else:
            return alpha

def fun(val):
    return val[0] ** 2 + 3 * val[1] ** 2
            #2x1+6x2

print (armijo(fun, [1, 2], [-1, -1]))