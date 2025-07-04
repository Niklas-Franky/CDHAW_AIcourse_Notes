import pandas as pd
import numpy as np
from torch import prelu

df = pd.read_csv('ai_course_data\\analysis.data')

#创建pandas专属的数据结构对象
#Series   DataFrame

#series存储一组数据和一组对应的索引
# s1 = pd.Series(['王宏',19],index=['姓名','年龄'])
# print(s1)
# s2 = pd.Series(np.arange(3,6))
# print(s2)
# x = np.random.randint(0,100,4)
# s3 = pd.Series(x,index=['A','B','C','D'])
# print(s3)
#注意输出中的dtype，int和object  后者是对象或者叫类
#获取数据集中的数值和索引
s = df.values
index = df.index
#索引  输入的是索引的各种形式，返回里面的数值或存储的其他东西
s4 = pd.Series([4,7,-5,3],index=['a','b','c','d'])
# print(s4[2]) # 索引
# print(s4[[0,3]]) # 索引列表
# print(s4[0:3]) # 切片
# 利用index名称
# print(s4['c']) # 以键取值
# print(s4['a']) # 名称索引
# print(s4[['a','d']]) # 名称索引列表
# print(s4['a':'d']) # 名称切片
#赋值
# s4[1]=8 # 修改一个值 或 s4['b']=8
# s4[2:]=3 # 修改多个值 或 s4['c':]=3
#其中，索引和切片可以是序号，也可以 是名称
#索引改名
s4.index=['Bob','Steve','Jeff','Ryan']
#注意：不允许对Series索引对象中的元素做修改。




#dataframe
#series的数值只有一列，故只用到行索引
#dataframe就有行、列两个索引，数据是表格结构
df1 = pd.DataFrame( [['sdfd',20,100],
                     ['xcvxcv',19,150],
                     ['erer',21,200]],
                   index = ['2252883','2252884','2252885'],
                   columns=['name','age','score'] )
                    #直接以2维数组创建dataframe，手动指定index和coloms
print(df1)

df2=pd.DataFrame({ 'num':['1951027','1753019','1850012','2051034'],
                   'name':['张莉','李峰','童敏','吴峰'],
                   'age':[19,20,20,18] } )
                #用字典的形式创建，键为colums,值为数据本身，index是默认
                #注意，这种写法值的格式很多样，因为是一维数据，可以是列表，元组，
                #也可以是一维的numpy数组，或者前面创建的series对象


#属性表 常见 需要记住
# <df名>.size 查看数据框元素的总个数，返回整数
# <df名>.shape 查看数据框的形状，返回元组
# <df名>.index 查看数据框的行索引, 返回RangeIndex对象
# <df名>.columns 查看数据框的列标签，返回Index对象
# <df名>.values 查看数据框中的数据，返回二维数组
# <df名>.info() 查看数据框的更多数据信息
# 可利用DataFrame对象的属性或方法查看其信息
# DataFrame三要素：columns、index 和 values

   
