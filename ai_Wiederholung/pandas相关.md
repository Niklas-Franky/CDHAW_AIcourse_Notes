```
import pandas as pd
df = pd.read_csv('ai_course_data\wine.csv')


df.head(n) 看前n行（常用于快速预览数据）
df.tail(n) 看后n行
df.shape 数据的行数、列数
df.columns 所有列名
df.index 所有行的索引
df.info() 数据基本信息（类型、空值等）
df.describe() 数值型数据的统计描述（均值、方差等）
df.values 直接转成numpy数组
df.loc[行名, 列名] 按标签选数据（比如df.loc[2, 'Alcohol']）
df.iloc[行下标, 列下标] 按下标选数据（比如df.iloc[2, 1]）
df['某列'] 选某一列（Series对象）
df.isnull() 判断哪些值是空的
df.drop(列名或行名, axis=0或1) 删行/列
df.groupby('列名') 分组
df.mean()、df.sum()、df.max()、df.min() 各种统计

```
