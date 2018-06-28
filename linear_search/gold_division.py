from Function import *


def gold_division(_f, val, d, dur):
    eps = 0.001

    def _fi(_alpha):
        return _f(array(val) + _alpha * array(d))

    fi = Function(_fi)
    lamb = dur[0] + 0.382 * (dur[1] - dur[0])
    mu = dur[0] + 0.618 * (dur[1] - dur[0])
    fi_lamb = fi.value(lamb)
    fi_mu = fi.value(mu)
    while True:
        if fi_lamb > fi_mu:
            if dur[1] - lamb < eps:
                return mu
            else:
                dur[0] = lamb
                lamb = mu
                mu = dur[0] + 0.618 * (dur[1] - dur[0])
        else:
            if mu - dur[0] < eps:
                return lamb
            else:
                dur[1] = mu
                mu = lamb
                lamb = dur[0] + 0.382 * (dur[1] - dur[0])
