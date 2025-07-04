import numpy as np
import pandas as pd

df = pd.read_csv('ai_course_data\\iris.data')


# 可通过索引器、切片和布尔索引等多种方法，选取行数据
# 获取行数据 说明
# <df名>.head(n) 查看前n行的数据，n默认取5。
# <df名>.tail(n) 查看后n行的数据，n默认取5。
# <df名>.loc[indexlist,collist] 根据(名称)索引列表/切片，查询某些行、某些列
# 的数据。省略collist，查询所有列。
# <df名>.iloc[iloclist,cloclist] 根据位置序号列表/切片，查询某些行、某些列的
# 数据。省略cloclist，查询所有列。
# <df名>[start:stop:step] 利用行切片获取行子集（所有列），start、stop、
# step都是整数（位置序号）
# <df名>[布尔索引] 利用布尔索引获取满足条件的行（所有列）。
# <df名>.loc[condition,collist]

# print(df.head(4),df.tail(5))

# df.loc['a']
# df.loc[['b','d']]
# df.loc['b':'d'] #行名

# df.iloc[0]
# df.iloc[[1,3]]
# df.iloc[1:3]#行号

#布尔索引
result = df[(df['sepal_length'] > 5) & 
            (df['sepal_length'] < 5.5)]
print(result)
#注意写法的逻辑，两个布尔索引数组进行与，得新的索引，再进行索引查询
#结果输出仍是dateframe对象

#索引+切片混用 当然也是可以的
# df.loc[:,['num','age']]# 使用iloc获取列子集
# df.iloc[:,[0,2]] # 使用iloc获取列子集
# df.iloc[:,::2]

# # 问题2：如何获取年龄小于均值的学生学号和姓名？
# df.loc[ df['age'] < df['age'].mean() , ['num', 'name']]
# # 问题1：如何获取所有学生的学号和年龄？
# df[['num','age']]
#注意这些方法跟在原df后面是可以改变df形状的，所以有需要的话记得创建新df对象以免数据丢失


#增
# df['birthday']=['2000-1-1','2001-12-1','2002-10-9','2003-4-5']
#将列表或数组赋给某列时，其长度须跟数据框的长度一致，否则报错。
#若将一个Series赋给某列，将自动补齐，所有空位都将被填上缺失值NaN
#改
# df['birthday']=pd.to_datetime(df['birthday'])#修改dtype类型
# print(df['birthday'])
#删
# <df名>.drop(columns =[列名1,列名2,...], inplace=False)
#inplace为true则生成新数组，flase则作用于原数组
#改
# <df名>[n:n+1]=新数据列表
#这是修改第n行的数据
#替换
# <df名>.replace(A,B, inplace=False)        数据框中所有A都用B替换。
# <df名>.replace({列标签:A},B, inplace=False)  指定列中所有A都用B替换












