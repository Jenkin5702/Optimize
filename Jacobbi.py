from numpy import *
from Function import *


def Jacobbi(a, n):
    m = size(a)
    A = []
    for i in range(0, n):
        a = range(0, m)
        for j in range(0, n):
            a[j] = Function(a[i]).diffun(j).fun
        A = A + [a]
    return A

def Jacobbi_val(_a,val):
    m = size(_a)
    n=size(val)
    A = []
    for i in range(0, m):
        a = Function(_a[i]).grad(val)
        A = A + [a]
    return array(A)