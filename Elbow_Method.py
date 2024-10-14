import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


mfcc_csv_file = "selected.mfcc.csv"
selection = pd.read_csv(mfcc_csv_file, sep=';', dtype='float')

sse = []  # 保存每个 k 值对应的 SSE
k_values = range(58,70)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)  # 设定随机状态确保可重复性
    kmeans.fit(selection)  # 训练 k-means 模型
    sse.append(kmeans.inertia_)  # inertia_ 是 SSE，即簇内误差平方和

plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_method_3.png')
plt.show()