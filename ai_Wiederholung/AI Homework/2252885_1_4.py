import numpy as n
import pandas as pd
x=n.random.randint(0,100,(10,4))
print(x)
mean=n.mean(x,axis=0)
std=n.std(x,axis=0)
x=(x-mean)/std
print(x)