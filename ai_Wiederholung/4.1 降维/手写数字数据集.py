import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv('ai_course_data\mnist_train.csv',header=None)
X_train = np.array(df.iloc[:,1:].values)

pca = PCA(n_components=0.95)
#这个参数既可以填要保留的成分数，也可以填需要到多少方差
X_pca = pca.fit_transform(X_train) #降维并返回
print(f'成分数量：{pca.n_components_}')
print(f'成分累计方差贡献率{pca.explained_variance_ratio_.sum()}')


#压缩效果可视化
X_recover = pca.inverse_transform(X_pca)

plt.figure(figsize=(8,4))
#画布尺寸为8英寸*4英寸
for idx,X in enumerate((X_train[::2000],X_recover[::2000])):
    #遍历这两张表，但是是每隔2000个数据遍历一次，idx是序号，x是某张表的数据
    plt.subplot(1,2,idx+1)
    plt.title(['原始数据','压缩后数据'][idx])
    for row in range(4):
        for col in range(4):
            plt.imshow(X[row * 4 + col].reshape(28, 28), cmap="binary",
                        vmin=0, vmax=255, extent=(row, row + 1, col, col + 1))
            plt.axis([0, 4, 0, 4])
            plt.axis("off")

