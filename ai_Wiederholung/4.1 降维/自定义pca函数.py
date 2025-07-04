import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pca(x,topN):
    sc = StandardScaler()
    #创造一个做标准化的类
    x_std = sc.fit_transform(x)
    #计算均值与方差，然后返回一个返回的数据，是一个矩阵
    U,S,Vt = np.linalg.svd(x_std)
    # linalg这个函数就是用来进行矩阵的svd运算的，返回的仍然是矩阵
    w= Vt.T[:,:topN]
    #对生成的右奇异矩阵进行转置，然后取我要的前topn列这么多个特征
    x_pca = x_std.dot(w)
    #对标准化后的数据进行投影的矩阵计算，乘以一个w矩阵
    x_recon =sc.inverse_transform(x_pca.dot(w.T))
    #相当于乘了一个w和w的转置，得到的新矩阵在进行一次标准化处理返回
    #本来x_pca是二维平面中沿着一个方向差不多一维的数据，现在将其重新变成
    #一个二维的点阵，也就是回到原数据集的维度，保持矩阵的大小不变
    #但是会丢失一部分信息，所以只是近似原来的矩阵
    return x_pca,x_recon

rng= np.random.RandomState(100)
#随机数生成对象，随机数种子为100就表示每次的随机数都不一样，种子越大随机性越高
x=np.dot(rng.randn(200,2),rng.rand(2,2))
#随机生成一个一定范围的200个数据点
#randn(200,2)是生成的一个形状为(200,2)的样本列表，值是正态分布，均值0方差1
#rand()是生成我们指定形状的0-1内的随机数
#点乘后仅仅是一个随机的点阵、矩阵，值并不在0-1内，有正有负而且中心也不是0

low_x,recon_x = pca(x,1)
#返回降维后的矩阵和还原后的矩阵
plt.scatter(x[:,0],x[:,1],alpha=0.7)
plt.scatter(recon_x[:,0],recon_x[:,1],alpha=0.2)
#由于是二维数据，故可以把第一列作为x轴，第二列作为y轴，alpha是透明度
#元数据集x和降维并重构后的数据集recon_x都能正常描点，但是low_x不行，因为是一维的
plt.scatter(low_x,np.zeros_like(low_x),alpha=0.5)
#描点的方法是把low_x作为x轴，再创建一个形状与low_x一样的全0数组(同样一维)
#用scatter手动设置x轴和y轴就可以显示了



plt.axis('equal')
plt.grid()
plt.show()