from numpy import *
from equation_constraint import *

def active_set(G,g,a1,b1,a2,b2):
    m1=a1.shape[0]
    m2=a2.shape[0]
    n=a1.shape[1]
    E=arange(0,m1)
    I=arange(m1,m1+m2)
    M=arange(0,n)
    D=setdiff1d(M,E)
    for i in range(m1,n):
        c=zeros(n)
        c[i]=1
        a1=vstack((a1,c))
    b1=concatenate([b1,D])
    x=dot(linalg.inv(a1),b1)[E]
    while True:
        if min(dot(a2,x)-b2)>=0:
            break
        else:
            b1[D]=b1[D]+ones(n-m1)
            x = dot(linalg.inv(a1), b1)[E]

    A=vstack((a1,a2[dot(a2,x)==0]))
    g_k=dot(G,x)+g
    p=equation_constraint(G,g_k,A,zeros(A.size[0]))

