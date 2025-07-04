from re import X
import numpy as np

a= np.array(   [[1,2,3], [1,2,3], [3,5,6]] ,dtype=np.int32 )
#np.array专门将各种列表、元组、以及其他pandas的数据类型转化为numpy类型数组
#注意格式，我们传入的参数是一个等长的 元祖、列表嵌套的结构
#因此括号的格式可以不用太在意[(),()] ([],[]) ((),()) [[],[]]都是一样转化为numpy类型数组
#一定是嵌套，两层括号，就算只有一层也行 如[[1,2,3]]
#注意dtype的写法

#等距数组
b = np.arange(1,10,2,dtype=np.int32)
c = np.linspace(1,5,100,dtype=np.double)
# dtype可以不用写
# arange是间隔取数  linspace是按数目均分

#特殊数组 
d = np.ones( [3,2] )   #全1
e = np.zeros( [4,5] )   #全0
f = np.ones_like( [[2,3,4],[4,5,6],[4,5,8]]  ) #全变1
g = np.zeros_like( [[3,3,3],[5,7,9],[1,5,8]] ) #全变0

#随机数组
rng = np.random.RandomState(100) #很好用的随机数生成器类
# 讲义中是写的完整的类名 如np.random.rand(),并没有随机数种子的参数
# 这个生成器就是把随机数种子这个参数单独拿出来生命，再用这个生成器取创建其他列表就不用填随机数种子了
h = rng.rand( 3,2  ) #注意只能放形状  #0~1
i = rng.randint(0,100,[4,6])     #给定的范围 0~100
j = rng.randn(4,7)   #形状  正态分布，均值0方差1
l = rng.choice(100,8)   #numpy类型的一维数组
m = rng.uniform(-2,7,[3,4]) #给定范围 形状  float数组
n = rng.normal([[3,3,3],[3,3,3]]) 
#默认均值0方差1的float数组，但是我们得传入形状 ，如上
# 或者[1,2,3]是一维3列的数组。只给一个单单的数就是生成一个float



#数组的属性
A=[[1,2,3,4],[10,20,30,40]]#嵌套列表
A_array = np.array(A)#numpy类型数组
# print(A_array.shape) # 数组形状 (2, 4)
# print(A_array.size)  # 元素总个数  8
# print(A_array.ndim)  # 数组维度 输出2
# print(type(A),type(A_array)) 
# <class 'list>嵌套列表也是列表 <class 'numpy.ndarray'>numpy类型数组
# print(a.dtype) # 数组内元素的数据类型 输出 int32
# #dtype类型有很多种   np.int8/16/32/64/128/256
                #   np.float8/16/32/64/128/256
                #   np.double
                #   np.bool




#索引
y = rng.randint(0,100,[10,10])
##简单索引
# y[2,1] ,y[2][1] #两者等价
# y[2][2] = 7 #赋值
# ##切片 与 切片赋值
# y[1:6:2 , 2:9:3] #x[start1:stop1:step1, start2:stop2:step2, ... ]

y_new = rng.randn(10,1)
# print(y)
# print(y[::-1 , ::])
# print(y[:: , ::-1])
# print(y[::-1 , ::-1])
# #步长取负值就是翻转，此时尽量标准写法以免报错
# #步长为1时，可以选择一些简化写法
# print(y[:3])#前3行
# print(y[::][-2:]) #注意！第一个括号的[::]并没有任何用，还是在取最后两行
# print(y.T[-3:]) #转置以后再这么取才是取的列的切片 ！！取完以后结构是(3,10)一定注意

#切片索引混用
#尤其注意 ：是切片用的，索引我们直接给数字或者布尔值
y[:,3] #第三列的所有行，这么混用只取一列则会导致降维，变成numpy类型的一维数组
y[2,:] #第二行所有列，同理，一维
y[:2,:] #这种看似混用实际上都用了切片符，还是属于在二维的空间里操作，结果是只有一行的二维数组

#索引列表
#索引不止用数字，还可以用列表，或者数组(见下面的布尔索引)
y[ [1,3,5],3 ] #取第1,3,5行的第三列，组成新的数组，仍是2维(但只有一行)
y[ [-2,-4] ,:] = 10#索引可以和切片符一起用.也可以赋值

#布尔索引
# print(y<4)  # y<4 是一个布尔运算，返回只有 true和false 的数组
#             # 返回的值既是一个数组，也是一个索引的数组，布尔值是可以当做索引用的
# print(y[ y<4 ])    # 可以理解为我用了一个索引列表去取数，这个列表是一个布尔数组
# 注意！由于数值的随机性没法做到取完以后数组任然保持原来的大小。布尔索引做完返回的是一个一维的列表！！




#变形
# y=y.reshape([5,20])
# print(y)
#行矩阵、列矩阵变换
z = np.array([1,2,3]) #这是一维的，注意结构
# z_new1 = z.reshape((1,3)) # 行矩阵
# z_new2 = z.reshape((3,1)) # 列矩阵  变换以后成了二维的 注意结构
# z_new3 = z[: , np.newaxis] #利用切片的操作变成列矩阵
# z_new4 = z[np.newaxis , :] #行矩阵
# z.flatten() #抹平成一维数组
#拼接
# arr1 = np.array([[1,2,3],[4,5,6]])
# arr2 = np.array([[7,8,9],[10,11,12]])
# arr3 = np.concatenate(arr1 , arr2 , axis=1)#axis默认0，即沿0轴水平方向，给一个1就是竖直的1方向




# 运算
# arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# 1 / arr
# arr ** 0.5
# arr // 2 #这也是广播的应用，每个元素都要进行一次算术运算
#数组广播
# x = np.array([[1,2], [3,4]])
# v = np.array([10, 20])
# y = x + v   # 矩阵x的每行都加上一维数组v
# print(y) 
#内积 点积
# np.dot(x,y)
# x.dot(y)
#其他聚合函数
xx = rng.randint(0,100,[5,5])
print(xx)
# print(np.sum(xx))      # 计算数组所有元素之和
# print(np.sum(xx, axis=0))  # 计算每列元素之和.理解为沿横轴，每个步长计算一次sum，所以是列元素和
# print(np.sum(xx, axis=1)) # 计算每行元素之和
# print(np.sum(xx %2 ==0)) # 统计arr中偶数元素个数
# print((xx%2==0).sum()) # 另一种语法
#布尔数组还可以看做是1和0的形式，当然可以sum求和
# print(xx.min(),xx.min(axis=0),xx.min(axis=1))
# print(xx.mean())#算术平均数
# print(xx.std(),xx.var()) #标准差 方差
# print(xx.argmax(),xx.argmin())  #很重要，求最大元素的索引
# print(xx.cunsum(axis=1))#横轴向的累计和
# print(xx.cumprod(axis=1))#竖直轴向的累积
#排序
xx.sort(axis=0) #逻辑和sum一样，0是每一列排序，1是每一行排序
print(xx)
print(xx.argsort())#每一行都进行排序返回索引，故可以选择轴，也可flatten之后整体排序







# print( n )