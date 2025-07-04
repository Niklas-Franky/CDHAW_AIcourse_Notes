import dis
from itertools import count
from turtle import TPen
from cv2 import countNonZero
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import im

# df = pd.read_csv('ai_course_data\\iris.data')
# df_data = df.values #只取values部分

X_train = np.array([[158, 64],[170, 86],[183, 84], 
                    [191, 80], [155, 49], [163, 59], 
                    [180, 67], [158, 54], [170, 67]])
y_train = ['male', 'male', 'male', 'male', 
           'female', 'female', 'female', 'female', 'female']


#画图
# plt.rcParams["font.sans-serif"]=["SimHei"]
# plt.rcParams["axes.unicode_minus"]=False
# #这两句设置标题的和标签的中文乱码问题，不止在jupiter notebook里有
# plt.xlabel('身高')
# plt.ylabel('体重')
# plt.title('分性别的身高体重图')

#遍历
# for idx, x in enumerate(X_train):
#     plt.scatter(x[0],x[1],
#                 alpha=0.9,s=70,
#                 c='blue' if y_train[idx]=='male' else 'red',
#                 marker='x' if y_train[idx]=='male' else 'D')
# plt.grid()
# plt.show()


#计算距离
x_test = np.array([[175,75]])
distance = (np.sum( (X_train - x_test)**2 ,axis=1) )**0.5
#得到直线距离的一个一维数组
k=3
print(distance)
distance_arg = np.argsort(np.array(distance))
# print(distance)
# nearest_nieghbor = distance[:3,:]
# print(np.size(nearest_nieghbor))

nearest_neighbor_index = distance_arg[:3]
# print(nearest_neighbor_index)

nearest_neighbor_genders = np.take(y_train,nearest_neighbor_index)
#注意，上面生成的索引是一个numpy类型的一维数组，y_train本身只是一个列表
#故不能当做列表的索引，会报错，使用np.take()可以解决数据类型不同的问题，很方便


from collections import Counter
b = Counter(nearest_neighbor_genders)
print(b.most_common(1)[0],[0])