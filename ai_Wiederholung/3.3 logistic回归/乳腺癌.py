from optparse import Values
from networkx import rich_club_coefficient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from xgboost import train

cancer = load_breast_cancer() #bunch对象
X = cancer.data
y = cancer.target #均返回numpy类型数组

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)

# logis = LogisticRegression(C=1,random_state=42,max_iter=10000)
# logis.fit(X_train,y_train)

# print('训练得分:{:.3f}'.format(logis.score(X_train,y_train)))
# print('测试得分:{:.3f}'.format(logis.score(X_test,y_test)))

mean = np.mean(X,axis=0)
var = np.var(X,axis=0)

mean_max = np.max(mean)
mean_min = np.min(mean)

var_max = np.max(var)
var_min = np.min(var)

sdfasf = pd.DataFrame([[mean_max,var_max],
                       [mean_min,var_min]],
                    index=['最小','最大'],
                    columns=['特征均值','特征方差']    )
print(sdfasf)
