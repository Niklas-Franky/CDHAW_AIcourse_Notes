import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#预处理
df = pd.read_csv('ai_course_data\\weather.csv')
le = LabelEncoder()
df['Description'] = le.fit_transform(df['Description']) #分类变量编码
X = df.values[:,0] 
y = df.values[:,1:]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


#boosting模型训练
gbc = GradientBoostingClassifier(n_estimators=1000,max_depth=4,learning_rate=0.1,n_iter_no_change=10,random_state=42)
gbc.fit(X_train,y_train)

#打印得分










