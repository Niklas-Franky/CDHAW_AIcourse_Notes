import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
df = pd.read_csv(url , header=None)

df.to_csv('wine.csv',index=False,header=False)
#index是行号

print('数据已经保存为 wine.csv')