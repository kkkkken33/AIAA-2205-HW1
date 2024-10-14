from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import argparse
import pickle
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("model_file")
parser.add_argument("--feat_appendix", default=".csv")
args = parser.parse_args()

# 1. read all features in one array.
fread = open(args.list_videos, "r")
feat_list = []
label_list = []
df_videos_label = {}

# Load video names and events in dict
for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    
    # For videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
        feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
        label_list.append(int(df_videos_label[video_id]))

print("number of samples: %s" % len(feat_list))
y_unstandard = np.array(label_list)
X_unstandard = np.array(feat_list)
scaler = StandardScaler()
y = scaler.fit_transform(y_unstandard)
X = scaler.fit_transform(X_unstandard)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape)  #(57000, 28, 28)
# print(y_train.shape)  #(57000, 10)
# print(X_test.shape)    #(3000, 28, 28)
# print(y_test.shape)    #(3000, 10)


# # 定义参数空间
# param_space = {
#     'hidden_layer_sizes': [(100,)],
# }

# # 创建 MLPClassifier
# mlp =  MLPClassifier(
#         activation='relu',           # 激活函数
#         solver='adam',               # 优化算法
#         alpha=0.0001,                # L2惩罚参数
#         batch_size='auto',           # 批量大小
#         learning_rate='constant',     # 学习率策略
#         learning_rate_init=0.001,    # 初始学习率
#         max_iter=200,                # 最大迭代次数
#         random_state=42              # 随机种子
#     )

# # 使用 BayesSearchCV 进行调参
# opt = BayesSearchCV(mlp, param_space, n_iter=50, random_state=42, cv=3)
# opt.fit(X_train, y_train)

# # 输出最佳参数和得分
# print("最佳参数组合：", opt.best_params_)
# print("最佳得分：", opt.best_score_)
