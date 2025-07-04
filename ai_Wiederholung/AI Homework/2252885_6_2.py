import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

file_path = r"F:\python-training\ai_course_data\ad.data"  
data = pd.read_csv(file_path, header=None)

data.replace(to_replace=r'\?', value=-1, regex=True, inplace=True)

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

y = y.map({'ad.': 1, 'nonad.': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'max_depth': [140, 150, 155, 160],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2, 3]
}

clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("最优参数组合：", grid_search.best_params_)

y_pred = best_model.predict(X_test)
print("分类性能报告：")
print(classification_report(y_test, y_pred))