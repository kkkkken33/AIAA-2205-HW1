import numpy as np

# 假设这是你的数据
data = np.array([6, 6, 6, 8, 8, 8])

# 将一维数组重塑为二维数组
data_reshaped = data.reshape(-1, 1)

# 现在可以传入 scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_reshaped)

print("归一化后的数据：")
print(normalized_data)