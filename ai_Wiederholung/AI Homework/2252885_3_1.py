import pandas as pd
import numpy as np
iris=pd.read_csv(r'C:\Users\22128\Desktop\3.1\iris\iris.data',names=['sepal_length','sepal_width','petal_length','petal_width','species'])
data=iris.values
# 定义 KNN 分类器 (k=1)
def knn_predict(X_train, y_train, X_test, p):
    predictions = []
    for test_point in X_test:
        # 计算每个训练点与测试点之间的闵可夫斯基距离
        distances = [distance.minkowski(test_point, x_train, p) for x_train in X_train]
        
        # 找到最近邻（k=1）
        nearest_index = np.argmin(distances)
        predictions.append(y_train[nearest_index])
    
    return np.array(predictions)

# 计算分类准确率
def evaluate_knn(X_train, y_train, X_test, y_test, p):
    y_pred = knn_predict(X_train, y_train, X_test, p)
    return accuracy_score(y_test, y_pred)


num_test=30
np.random.seed(3)
np.random.shuffle(data)
test_set = data[:num_test]#定义测试集，取前30个样本
Xtest =np.array(test_set[:,:4],dtype='float')
ytest =np.array(test_set[:,-1])
train_set = data[num_test:]#定义训练集，取后120个样本
Xtrain=np.array(train_set[:,:4],dtype='float')
ytrain=np.array(train_set[:,-1])

# 比较不同 p 值的准确率
p_values = range(1, 8)
accuracies = []

for p in p_values:
    accuracy = evaluate_knn(X_train, y_train, X_test, y_test, p)
    accuracies.append(accuracy)
    print(f"p={p}, Accuracy: {accuracy:.4f}")

# 找出最优的 p 值
best_p = p_values[np.argmax(accuracies)]
print(f"Best p: {best_p}")