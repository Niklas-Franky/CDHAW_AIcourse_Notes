import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
data = pd.read_csv(r"C:\Users\22128\Desktop\AI作业\3.2\weather.csv")
X = data.drop(columns=['Description'])
y = data['Description']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
result=k=0
for i in range(1,20):
    model=KNeighborsClassifier(i)
    scores=cross_val_score(model,X_train,y_train,cv=5)
    if scores.mean()>result:
        result=scores.mean()
        k=i

print(f"最优k值为: {k}")
model= KNeighborsClassifier(k)
model.fit(X_train, y_train)
y_pred =model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("分类准确率:" ,accuracy)
report = classification_report(y_test, y_pred)
print("分类性能报告:")
print(report)