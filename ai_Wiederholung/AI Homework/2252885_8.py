from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#降维
pca=PCA(n_components=2)
pca.fit(X_train_std)
X_train_pca=pca.transform(X_train_std) #降维后
X_test_pca = pca.transform(X_test_std)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], hue=y_train, style=y_train, 
markers=['o', 's', '^'], s=100)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')


#logistic回归
lr = LogisticRegression(random_state=42)
lr = lr.fit(X_train_pca, y_train)
print(f'训练得分{lr.score(X_train_pca,y_train):.4f}')
print(f'测试得分{lr.score(X_test_pca,y_test):.4f}')