import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\22128\Desktop\AI作业\3.3\sh_air_quality_1.csv",encoding='GBK')
X=df[['AQI指数', 'PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3']]
y=df['质量等级']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
result=k=0
for i in range(1,20):
    model=KNeighborsClassifier(i)
    cv_result=cross_val_score(model,X_train,y_train,cv=5)
    if cv_result.mean()>result:
        result=cv_result.mean()
        k=i
print(k,result)
plt.plot(X,y)
bestmodel=KNeighborsClassifier(i)
bestmodel.fit(X_train,y_train)
y_pred = bestmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("准确率:" ,accuracy)