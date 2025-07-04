import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#例3.6多项式的回归以及学习曲线的绘制
根据 train_sizes 取出 1% 到 100% 的训练集，共取 40 个点（递增）
对于每一个训练集大小，都会做一次交叉验证（cv=5）
每轮的得分用 scoring 算（如 RMSE），分别记录：
训练得分（train_scores）
验证得分（valid_scores）