# from numpy import *
from Function import *
from unconstrained.descent import *


def barrier(f, a2, x0):
    mu = 0.1
    eps = 0.05
    x = array(x0)

    def p(_mu):
        def _q(x_):
            r = f(x_)
            for i in range(0, size(a2)):
                a = a2[i]
                r -= _mu * log(a(x_))
            return r

        return _q

    while True:
        p_mu = p(mu)
        Q = Function(p_mu)
        x = descent(p_mu, x.tolist())
        if Q.norm(x) < eps:
            return x
        else:
            mu = mu / 2


def f(val):
    return val[0] - 2 * val[1]


def a21(val):
    return 1 + val[0] - val[1] ** 2


def a22(val):
    return val[1]


print(barrier(f, [a21, a22], [3,1]))
