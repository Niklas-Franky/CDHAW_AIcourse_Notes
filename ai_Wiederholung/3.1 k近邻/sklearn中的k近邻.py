import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN



df = pd.read_csv('ai_course_data\iris.data')
df_data = df.values
X_data = df_data[:,0:4]
y_data = df_data[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,
                test_size=30)
#数据处理 分割


#sklearnh中的knn类                
# knn = KNN(n_neighbors=3)
# knn.fit(X_train,y_train)
# print(f'测试集分类准确率{knn.score(X_test,y_test)}')



#sklearn中的交叉训练评分函数
from sklearn.model_selection import cross_val_score
# cvs_result = cross_val_score(knn,X_train,y_train,cv=6)
# a = np.array(cvs_result)
# # 交叉验证是提高训练得分的，本身并不是一个像knn一样的估计器模型
# # 输入要用的模型 knn ， 训练集  ，交叉的折数
# print(f'交叉得分{cvs_result}')
# print('5折交叉的得分平均值：%.4f'  %cvs_result.mean()   )




#用交叉验证找到最优的k值并重新估计
result = []#空的列表，用来存存放最后的每个k与对应的交叉验证平均分
for k in range(1,40,2):
    knn = KNN(n_neighbors=k)
    cvs_result = cross_val_score(knn,X_train,y_train,cv=5)
    result.append(cvs_result.mean()) #填数据，不填列表元素
result = np.array(result)
print(type(result),result.dtype)
print(result)
best_result_index = result.argmax()
best_k = 2*(best_result_index+1)-1
print(best_k)
#讲义上关于result的排序做了一点修改，使用更简单的函数达到同样的目的

best_knn = KNN(n_neighbors=best_k)
best_knn.fit(X_train,y_train)
test_score = best_knn.score(X_test,y_test)

print('k=%d 训练集测试得分为：%.4f' %(best_k,test_score)  )
