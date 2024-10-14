import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import os
import xgboost as xgb
from sklearn.neural_network import MLPClassifier



parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':
    args = parser.parse_args()

    fread = open(args.list_videos, "r")
    feat_list = []
    label_list = []
    df_videos_label = {}
    for line in open(args.list_videos).readlines()[1:]:
        video_id, category = line.strip().split(",")
        df_videos_label[video_id] = category

    for line in fread.readlines()[1:]:
        video_id = line.strip().split(",")[0]
        feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
        if os.path.exists(feat_filepath):
            feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
            label_list.append(int(df_videos_label[video_id]))

    print("number of samples: %s" % len(feat_list))
    y = np.array(label_list)
    X = np.array(feat_list)

# 创建分层交叉验证对象
skf = StratifiedKFold(n_splits=10)

# 初始化分类器
# model = xgb.XGBClassifier(n_estimators=300)
model = xgb.XGBClassifier(
    n_estimators=300,
    # device = 'cuda'
)

# model = MLPClassifier(
#     hidden_layer_sizes=(200, ),  # 一层包含100个神经元
#     activation='relu',           # 激活函数
#     solver='adam',               # 优化算法
#     alpha=0.0001,                # L2惩罚参数
#     batch_size='auto',           # 批量大小
#     learning_rate='constant',     # 学习率策略
#     learning_rate_init=0.002,    # 初始学习率
#     max_iter=200,                # 最大迭代次数
#     random_state=42              # 随机种子
# )

# 存储每次折叠的准确率
accuracies = []

# 分层交叉验证
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练模型
    model.fit(X_train, y_train)    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 输出每个折叠的准确率和平均准确率
print("每个折叠的准确率:", accuracies)
print("平均准确率:", np.mean(accuracies))
