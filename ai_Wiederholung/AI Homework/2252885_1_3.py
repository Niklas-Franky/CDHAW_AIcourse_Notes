import numpy as n
import pandas as pd
A=n.random.rand(100,2)
print(A)
x = n.random.choice(A.shape[0],10,replace=False)#索引
B=A[x]
print(B)

i=0
while i<10:
    selected=B[i]
    dist=n.sum((A-selected)**2,axis=1)
    idx=n.argsort(dist)
    neighbor=A[idx[1:2]]
    print(neighbor)
    i+=1
