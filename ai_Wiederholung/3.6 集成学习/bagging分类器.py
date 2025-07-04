import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sympy import primitive
#作为分类器的评分参数


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

#训练分类器 并拟合
tree = DecisionTreeClassifier(random_state=1)
bag = BaggingClassifier(estimator=tree,n_estimators=500,max_samples=0.8,n_jobs=-1)

tree.fit(X_train,y_train)
bag.fit(X_train,y_train)

#预测新数据  
X_pred_tree = tree.predict(X_test)
X_pred_bag = bag.predict(X_test)

#训练集上的得分
print('单棵决策树训练集上的得分为：{}'.format(tree.score(X_train,y_train)))
print('bagging在训练集的得分：{}'.format(bag.score(X_train,y_train)))

#测试集上的得分 accuracy_score
print('单棵决策树在测试集上的准确率：{:.4f}'.format(X_pred_tree,y_test))
print('bagging在测试集上的准确率：{:.4f}'.format(X_pred_bag,y_test))