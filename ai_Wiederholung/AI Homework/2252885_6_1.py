import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


file_path = r"F:\python-training\ai_course_data\buyCar.csv"  # 替换为您的文件路径
data = pd.read_csv(file_path)

data = data.drop(columns=["CustomerID"])

data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1})
data["Income"] = data["Income"].map({"Low": 0, "Middle": 1, "High": 2})
data["Bought"] = data["Bought"].map({"No": 0, "Yes": 1})

X = data[["Age", "Sex", "Income"]]
y = data["Bought"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=["Age", "Sex", "Income"], class_names=["No", "Yes"], filled=True)
plt.show()

new_data = [[28, 0, 2]]  
prediction = clf.predict(new_data)
prediction_label = "Yes" if prediction[0] == 1 else "No"


y_pred = clf.predict(X_test)
classification_report_output = classification_report(y_test, y_pred)

# 输出结果
print("新数据预测结果:", prediction_label)
print("\n模型分类性能报告:\n", classification_report_output)