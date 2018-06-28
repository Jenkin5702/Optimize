from numpy import *

from Function import *


def wolfe(_f, val, d):
    a = 0.0
    b = 10000.0

    def _fi(_alpha):
        return _f(array(val) + _alpha * array(d))

    fi = Function(_fi)
    f = Function(_f)

    fi1 = f.value(val)
    g=f.grad(val)
    dfi1 = dot(g,array(d).T)
    dfi0 = fi.diff(0)
    alpha = 1.0
    i=0
    while True:
        i=i+1
        if i>30:
            return alpha
        fi_=fi.value(alpha)
        if fi_-fi1>1/3*alpha*dfi1:
            alpha_=a+(alpha-a)/2/(1+(fi1-fi_)/(alpha-a)/dfi1)
            b=alpha
            alpha=alpha_
        else:
            dfi_alpha=fi.diff(alpha)
            if dfi_alpha<1/2*dfi1:
                alpha_=alpha+(alpha-a)*dfi_alpha/(dfi1-dfi_alpha)
                a=alpha
                fi1=fi_
                dfi1=dfi_alpha
                alpha=alpha_
            else:
                return alpha