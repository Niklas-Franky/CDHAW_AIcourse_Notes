import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("creditcard.csv")

X = df.drop(["Time", "Class"], axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_resampled, y_resampled)

y_pred = rf.predict(X_test)
print("\n📋 分类报告（关注欺诈交易的 recall）:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正常', '欺诈'], yticklabels=['正常', '欺诈'])
plt.title("混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.tight_layout()
plt.show()

importances = rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔥 特征重要性前10：")
print(feat_imp_df.head(10))
