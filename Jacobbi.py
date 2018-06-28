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

# def a11(x):
#     return x[0]+x[1]**2-x[2]
#
# def a12(x):
#     return x[0]**2-x[1]+x[2]**2
#
# print(Jacobbi_val([a11,a12],[3,1,2]))