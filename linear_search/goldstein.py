from numpy import *

from Function import *


def goldstein(f, val, d):
    a = 0
    b = 10000
    alpha = 1

    def _fi(_alpha):
        return f(val + _alpha * array(d))

    fi = Function(_fi)
    fi0 = fi.value(0)
    dfi0 = fi.diff(0)
    while True:
        fi_alpha = fi.value(alpha)
        if fi_alpha > fi0 + 1 / 3 * alpha * dfi0:
            b = alpha
            alpha = (a + b) / 2
        elif fi_alpha < fi0 + 2 / 3 * alpha * dfi0:
            a = alpha
            if b == 10000:
                alpha = 2 * alpha
            else:
                alpha = (a + b) / 2.
        else:
            return alpha



def fun(val):
    return val[0] ** 2 + 3 * val[1] ** 2
            #2x1+6x2

print (goldstein(fun, [1, 2], [-1, -1]))