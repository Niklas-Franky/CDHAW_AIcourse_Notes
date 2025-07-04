import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding


#生成瑞士卷并可视化
X_swiss ,t = make_swiss_roll(n_samples=1000,noise=0.2,random_state=100)
#生成一系列三维数据点，t是有关着色的一个参数
darker_hot = ListedColormap(plt.cm.hot(np.linspace(0, 0.8, 256)))
#np.linspace(0, 0.8, 256)等距数组，0-0.8均分256份 [0.0, 0.003137, 0.006275, ..., 0.8]
#plt.cm.hot是matplot自带的一个色带，火红的颜色，从黑-红-黄
#plt.cm.hot作用是一个映射函数，输入等距数组，
#返回每个元素相对应的颜色(本质上是RGBA颜色的数组形式 如[0.0, 0.0, 0.0, 1.0])
#ListedColormap是创建了一个"整体颜色映射器"的类，相当于做了一个手册出来，plt
#想画图的时候，那1000个点就能按照这里面的顺序进行着色，作为参数cmap使用

axes = [-11.5, 14, -2, 23, -12, 15]#这玩意没用到
fig = plt.figure(figsize=(6, 5))
#建一个二维的6*5的画布
ax = fig.add_subplot(111, projection='3d')
# 给画布加一个subplot也即三维坐标系
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], 
c=t, cmap=darker_hot) #注意cmap参数

ax.view_init(10,-70) # 设置仰角为10度，方位角为-70度，
ax.set_xlabel("$x_1$", labelpad=6, rotation=0)
ax.set_ylabel("$x_2$", labelpad=6, rotation=0)
ax.set_zlabel("$x_3$", labelpad=6, rotation=0)
#这是数学公式的字体，也即latex，不用管
plt.show()



#LLE展开瑞士卷
lle = LocallyLinearEmbedding(n_neighbors=10,random_state=100)
X_unrolled = lle.fit_transform(X_swiss)

plt.scatter(X_unrolled[:,0],X_unrolled[:,1],c=t,cmap=darker_hot)
plt.xlabel('$z_1$')
plt.ylabel('$z_2$',)
plt.axis([-0.055, 0.060, -0.070, 0.090])
plt.grid()
plt.show()

#显示z1和t的相关性
plt.title("$z_1$ vs $t$")
plt.scatter(X_unrolled[:, 0], t, c=t, cmap=darker_hot)
plt.xlabel("$z_1$")
plt.ylabel("$t$", rotation=0)
plt.grid(True)
plt.show()
