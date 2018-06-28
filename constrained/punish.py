# from numpy import *
from Function import *
from unconstrained.simu_newton import *


def punish(f, a1, a2, x0):
    mu = 0.5
    eps = 10 ** (-8)
    x = array(x0)

    def q(_mu):
        def _q(x_):
            r = f(x_)
            for i in range(0, size(a1)):
                a = a1[i]
                r += 0.5 / _mu * a(x_) ** 2
            for i in range(0, size(a2)):
                r = r + 1 / 2 / _mu * (min(a2[i](x_), 0)) ** 2
            return r

        return _q

    while True:
        q_mu = q(mu)
        Q = Function(q_mu)
        x = simu_newton(q_mu, x.tolist())
        print (x)
        if Q.norm(x) < eps:
            return x
        else:
            mu = mu / 2


def f(val):
    return (val[0]-1)**2 + (val[1]-1)**2


def a11(val):
    return val[0]+val[1]-1


print(punish(f, [a11], [], [4., -5.]))
