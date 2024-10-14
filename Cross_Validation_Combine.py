import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle

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
label_list = {}
df_videos_label = {}

# Load video names and events in dict
for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = int(category)  # 保存为整数

for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    
    # For videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
        feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

print("number of samples: %s" % len(feat_list))
y = np.array(list(df_videos_label.values()))
X = np.array(feat_list)

# 2. Split the dataset into training and validation sets
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# 创建一个空数组来存储最终的预测
final_stacked_predictions = []

i = 0
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    i += 1
    # Train MLP model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(200, ),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.002,
        max_iter=200,
    )
    mlp_model.fit(X_train, y_train)

    # Train SVM model
    svm_model = SVC(probability=True)  # 需要probability=True以便于堆叠
    svm_model.fit(X_train, y_train)

    # 预测
    mlp_val_pred = mlp_model.predict_proba(X_val)  # 获取概率
    svm_val_pred = svm_model.predict_proba(X_val)  # 获取概率

    # 将MLP和SVM的预测概率结合
    combined_pred = np.hstack((mlp_val_pred, svm_val_pred))

    # 训练逻辑回归进行最终验证
    stacked_model = LogisticRegression(max_iter=1000, multi_class='ovr')
    stacked_model.fit(combined_pred, y_val)

    if(i == 5):
        # 保存模型
        print("Saving model...")
        pickle.dump(stacked_model, open(args.model_file, "wb"))
        print("Model saved.")
    # 进行最终预测
    final_val_pred = stacked_model.predict(combined_pred)

    # 计算准确率
    accuracy = accuracy_score(y_val, final_val_pred)
    accuracies.append(accuracy)
    print(f'Fold Accuracy (Stacked): {accuracy:.5f}')

# 计算平均准确率
average_accuracy = np.mean(accuracies)
print(f'Average K-Fold Accuracy: {average_accuracy:.5f}')
