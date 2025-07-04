import pandas as pd
df = pd.read_csv('train.csv', index_col='PassengerId')
y = df['Survived']
X = df.drop('Survived', axis=1)
X = X.drop(['Name', 'Ticket', 'Cabin'], axis=1)
X = X.fillna(0)
X['Sex'] = X['Sex'].replace({'male': 0, 'female': 1})
X = pd.get_dummies(X, columns=['Embarked'], prefix='Embarked')
print("目标变量 y:\n", y.head())
print("\n特征矩阵 X:\n", X.head())