import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import argparse
import pickle
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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
y = np.array(label_list)
X = np.array(feat_list)

# 2. Split the dataset into training and validation sets
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)

    model = MLPClassifier(
        hidden_layer_sizes=(200, ),  # 一层包含100个神经元
        activation='relu',           # 激活函数
        solver='adam',               # 优化算法
        alpha=0.0001,                # L2惩罚参数
        batch_size='auto',           # 批量大小
        learning_rate='constant',     # 学习率策略
        learning_rate_init=0.002,    # 初始学习率
        max_iter=200,                # 最大迭代次数
        random_state=42              # 随机种子
    )
    model.fit(X_train, y_train)


    y_val_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_val_pred)
    accuracies.append(accuracy)
    print(f'Fold Accuracy: {accuracy:.5f}')

# 6. Calculate average accuracy across all folds
average_accuracy = np.mean(accuracies)
print(f'Average K-Fold Accuracy: {average_accuracy:.5f}')