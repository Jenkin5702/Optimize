from numpy import *


def equation_constraint(G, g, A, b):
    m = min(A.shape)
    n = max(A.shape)
    M = range(0, n)
    E = range(0, m)
    A_B = A[E, :]
    I = setdiff1d(M, E)
    A_N = A[I, :]
    i = 0
    x = arange(0, n).astype(float)
    while True:
        if linalg.matrix_rank(A_B) == m:
            break
        else:
            i += 1
            E[m - i] += 1
            I = setdiff1d(M, E)
            A_B = A[E, :]
            A_N = A[I, :]
    G_BB = G[E, :][:, E]
    G_AA = G[I, :][:, I]
    G_AB = G[I, :][:, E]
    G_BA = G[E, :][:, I]
    g_B = g[E]
    g_A = g[I]
    invABAN = dot(A_N, linalg.inv(A_B))
    invABb = dot(linalg.inv(A_B).T, b)
    G_hat = G_AA - dot(G_AB, invABAN.T) - dot(invABAN, G_BA) + dot(invABAN, dot(G_BB, invABAN.T))
    g_hat = g_A - dot(invABAN, g_B) + dot(G_AB - dot(invABAN, G_BB), -invABb)
    x[E] = invABb + dot(dot(invABAN.T, linalg.inv(G_hat)), g_hat)
    x[I] = -dot(linalg.inv(G_hat), g_hat)
    return x


G = array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
g = array([-8, -3, -3])
A = array([[1, 0], [0, 1], [1, 1]])
b = array([3, 0])
print(equation_constraint(G, g, A, b))
