import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\22128\Desktop\AI作业\2.2\outlier.csv")
def detect_outliers(data):
    z_scores = (data - data.mean()) / data.std()
    return z_scores.abs() > 3
outliers = detect_outliers(df)
df.loc[outliers['english'], 'english'] = 90
df['computer'] = df['computer'].mask(outliers['computer'], df['computer'].mean())
print(df)