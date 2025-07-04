import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rng = np.random.RandomState(100) #随机数生成器
std = StandardScaler() #标准化类
x = np.dot(rng.randn(200,2),rng.rand(2,2)) 
x_std = std.fit_transform(x) 

pca = PCA( n_components=1) #sklearn中的pca生成器
x_pca = pca.fit_transform(x_std)
recon_x = pca.inverse_transform(x_pca)
#注意！这里的recon矩阵只是把数据逆变回了原来的空间，只是逆变换重构
#自定义pca函数中我们还做了标准化，所以是逆标准化重构
#如果还想逆标准化，则可以recon_x=std.inverse_transform(recon_x),这样就和元数据处于同一空间

plt.scatter(x[:,0],x[:,1],alpha=0.2)
plt.scatter(recon_x[:,0],recon_x[:,1],alpha=0.8)
plt.scatter(x_pca,np.zeros_like(x_pca),alpha=0.5)
plt.grid()
plt.show()

print("第一主成分占方差比：",pca.explained_variance_ratio_)