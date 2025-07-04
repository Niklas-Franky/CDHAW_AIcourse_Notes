import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

df = pd.read_csv('ai_course_data\\weather.csv')
X = df.values[:,0] 
y = df.values[:,1:]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
n= X_train.shape[0]*0.2
X_val = X_train[:n]
y_val = y_train[:n] #验证集
X_train = X_train[n:]
y_train = y_train[n:] #训练集 只占总的0.6


xgbr = XGBRegressor()













