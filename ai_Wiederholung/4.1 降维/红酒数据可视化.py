import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df_wine = pd.read_csv('ai_course_data\wine.csv') 
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']


X,y =df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
# iloc是Pandas专门用于按位置（下标）选数据的属性。
# 比如df.iloc[2, 3]（取第3行第4列）| df.iloc[:, 1:3]（取所有行，第2和第3列）
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=100)

std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)
#我们已经知道标准化是减去均值除以标准差
#一个标准化类只能有一个均值和标准差参数，fit_transform把训练集进行了“标准化计算”
#已经得到了这两个参数，那么测试集就不用再fit，直接transform就行了



#方法一：利用svd算法,求解前两个特征
# U,S,Vt = np.linalg.svd(X_train)
# w = Vt.T[:,:2]
# X_train_pca = X_train_std.dot(w)

#     #画图
# colors = ['r','b','g']
# markers = ['o','s','^']
# for l,c,m in zip(np.unique(y_train),colors,markers):
# #unique是去重
# #这里的for循环是按类别循环的，zip函数将3中label和按顺序排列好的color和marker一一对应
# #按类别循环时，就会用到下面的布尔索引筛选出对应的类别，看似只有3个类别循环3次其实是对每个数据的类别都进行一次循环 
# #重点理解按类别循环和按序号循环的区别
#     plt.scatter(X_train_pca[y_train==l,0],
#                 #y_train==l布尔索引，筛选出label满足l的那些数据
#                 #0表示取这些数据的第一项特征，1就是第二项特征，分别作为x/y轴
#                 X_train_pca[y_train==l,1],
#                 c=c,marker=m,label = f'class{l}')

# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc = 'lower left')
# plt.grid()
# plt.show()


#方法二 利用sklearn中的PCA类求解
pca = PCA(n_components=2)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)

    #画图
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=f'Class {l}', marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.grid()
plt.show()


X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression(random_state=1)
lr = lr.fit(X_train_pca, y_train)
#只有特征部分需要降维，label也就是目标值本来就是一维数组，不需要降维
#用模型拟合时记得把测试集也降维处理
print(f'训练集得分：{lr.score(X_train_pca,y_train):.4f}')
print(f'测试集得分：{lr.score(X_test_pca,y_test):.4f}')
#注意写法 f'   {写外部参数}'
#  :.4f   紧跟在要处理的参数后面，代表取4位小数
print('第一第二主成分占方差比',pca.explained_variance_ratio_)



