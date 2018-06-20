# coding=utf-8
from numpy import *


def simple(c, A, b):
    m = min(A.shape)
    n = max(A.shape)
    M = range(0, n)
    E = range(0, m)
    B = A[:, E]
    I = setdiff1d(M, E)
    N = A[:, I]
    i = 1
    x = arange(0, n).astype(float)
    while True:
        if linalg.matrix_rank(B) == m:
            x[E] = dot(linalg.inv(B), b).flatten()
            x[I] = arange(0, n - m)
            if min(x) >= 0:
                break
        E[m - i] += 1
        I = setdiff1d(M, E)
        N = A[:, I]
        B = A[:, E]
        i = i + 1
    # -------
    # E=[2,3,4]
    # I=[0,1]
    # N = A[:, I]
    # B = A[:, E]
    # x=array([0.,0.,3.,2.,16.])
    # -------
    cb = c[E]
    cn = c[I]
    # 单纯形乘子
    y = dot(linalg.inv(B.T), cb)
    # 简约价值系数
    cv = cn - dot(N.T, y)
    eps = 10 ** (-7)
    r = b
    while min(cv) < 0:
        p = I[where(cv == min(cv))[0][0]]
        ap = dot(linalg.inv(B), A[:, p])
        bq = r[ap > eps] / ap[ap > eps]
        bq = min(bq)
        q = E[where(r / ap == bq)[0][0]]
        out = where(array(E) == q)[0][0]
        enter = where(array(I) == p)[0][0]
        E[out] = p
        I[enter] = q
        N = A[:, I]
        B = A[:, E]
        x[E] = dot(linalg.inv(B), b).flatten()
        r = x[E]
        x[I] = zeros(n - m)
        cb = c[E]
        cn = c[I]
        y = dot(linalg.inv(B.T), cb)
        cv = cn - dot(N.T, y)
    print(x)


c = array([-2, -3, 0, 0, 0])
A = array([[-1, 1, 1, 0, 0], [-2, 1, 0, 1, 0], [4, 1, 0, 0, 1]])
b = array([3, 2, 16])
simple(c, A, b)
