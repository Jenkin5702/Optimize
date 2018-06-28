from linear_search.wolfe import *
from Jacobbi import *
from square_planning.equation_constraint import *


def solve(_A, _b):
    A = array(_A)
    n = A.shape[0]
    m = A.shape[1]
    for i in range(n, m):
        v = zeros(m)
        v[i] = 1
        A = vstack((A, v))
    return dot(linalg.inv(A), array(_b))


def sequence(_f, _x, a2):
    x = array(_x)
    m = x.size
    X = range(0, m)
    n = size(a2)
    N = range(0, n)

    def _l(val):
        value = array(val)[X]
        r = _f(value)
        for i in N:
            a = a2[i]
            r -= val[i] * a(value)
        return r

    def fi(mu):
        def _fi(x, mu):
            r = _f(x)
            for a in a2:
                r += abs(a(x))
            return r

        return _fi

    f = Function(_f)
    l = Function(_l)

    while True:
        A = Jacobbi_val(a2, x)

        g = f.grad(x)
        lamb = solve(A, g)
        G = l.hesse(concatenate([x, lamb]).tolist())[X, :][:, X]

        c = range(0, n)
        for i in range(0, n):
            a = a2[i]
            c[i] = a(x)

        dx = equation_constraint(G, g, A, c)
        mu = 2
        fi_x = fi(mu)
        if linalg.norm(c) + linalg.norm(g - dot(A.T, lamb)) < 0.01:
            return x
        alpha = wolfe(fi_x, x, dx)
        x = x + alpha * dx


def f(val):
    return 2 * val[0] + 3 * val[1]


def a1(val):
    return val[0] ** 2 + val[1] ** 2


sequence(f, [-0.7, -0.7], [a1])
