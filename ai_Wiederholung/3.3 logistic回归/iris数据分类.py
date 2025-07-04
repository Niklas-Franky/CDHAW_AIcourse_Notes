import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ai_course_data\\iris.data')
df_value = df.values

X = df_value[:,2:4] #两个特征列
y = df_value[:,-1]  #目标数组

for idx,i in enumerate(y):
    if i == 'virginica':
        y[idx] == 1
    else:
        y[idx] == 0
#对分类变量进行的简单处理

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
logis = LogisticRegression(C=2,random_state=42)
logis.fit(X_train,y_train)
