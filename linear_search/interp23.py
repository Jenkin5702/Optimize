from numpy import *

def interp23(a1, a2, a3, _f, d):
    def _fi(_alpha):
        return _f(array(val) + _alpha * array(d))

    fi1 = fun(a1)
    fi2 = fun(a2)
    fi3 = fun(a3)
    k = 0
    ap = 0
    for x in np.arange(range):
        b = ((a2 ** 2 - a3 ** 2) * fi1 + (a3 ** 2 - a1 ** 2) * fi2 + (a1 ** 2 - a2 ** 2) * fi3) / (
        (a1 - a2) * (a3 - a1) * (a2 - a3))
        a = -((a2 - a3) * fi1 + (a3 - a1) * fi2 + (a1 - a2) * fi3) / ((a1 - a2) * (a3 - a1) * (a2 - a3))
        if a != 0:
            ap = -b / (2 * a)
        else:
            ap = 0
        if ap > a2:
            if fun(ap) <= fun(a2):
                a1 = a2
                a2 = ap
            else:
                a3 = ap
        else:
            if fun(ap) <= fun(a2):
                a3 = a2
                a2 = ap
            else:
                a1 = ap
        k = k + 1
    print ("ap=:    -------", ap, "-------")
    return ap
