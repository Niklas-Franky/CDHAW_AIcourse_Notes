import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\22128\Desktop\AI作业\2.1\score_test.csv")
df.fillna(df.mean(axis=0),inplace=True)
df['总分']=df.iloc[:,1:].sum(axis=1)
def classify(score):
    if score>=85:
        return "A"
    elif score>=60:
        return "B"
    else:
        return "C"
df['grade']=df['总分'].apply(classify)
df.drop (columns=['总分'])
print(df)
df.to_csv(r"C:\Users\22128\Desktop\AI作业\2.1\score_test1.csv")