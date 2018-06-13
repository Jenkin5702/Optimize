from linear_search.goldstein import goldstein
from unconstrained.conjugate import *
def f(val):
    return 100*(val[0]**2-val[1])**2+(val[0]-1)**2
    # return val[0] ** 2 + 3 * val[1]

print (conjugate(f,[1,2]))