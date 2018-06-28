# from numpy import *
from Function import *
from unconstrained.descent import *


def barrier(f, a2, x0):
    mu = 0.2
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

    nr = 1.0
    while nr > 0.1:
        mu = mu / 2
        p_mu = p(mu)
        Q = Function(p_mu)
        x = descent(p_mu, x.tolist())
        print(x)
        nr = Q.norm(x)
    return x


def f(val):
    return 2 * val[0] + 3 * val[1]


def a21(val):
    return 1 - 2 * val[0] ** 2 - val[1] ** 2


def a22(val):
    return val[1]


print(barrier(f, [a21], [0.3, 0.2]))
