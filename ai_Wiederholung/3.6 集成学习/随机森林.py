import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('ai_course_data\\wine.csv',header=None)
df.columns = ['Class label','Alcohol',
            'Malic acid','Ash','Alcalinity of ash',
            'Magesium','Total phenols','Flavanoids',
            'Nonflavanoid phenols','Proanthocyanins',
            'Color intensity','Hue',
            'OD280/OD315 of diluted wines','Proline']
#数据拆分
X = df.values[:,1:]
y = df.values[:,0]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

#随机森林分类器
rf = RandomForestClassifier(n_estimators=500,max_samples=0.8,n_jobs=-1,random_state=1)

#训练数据
rf.fit(X_train,y_train)
a = rf.score(X_train,y_train)
y_pred = rf.predict(X_test)
b= accuracy_score(y_pred,y_test)

print('训练：{}'.format(a))
print('测试：{}'.format(b))
