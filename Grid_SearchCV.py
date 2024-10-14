from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import argparse

# 定义模型
mlp = MLPClassifier()

# 定义参数网格
param_grid = {
    'hidden_layer_sizes': [(500,), (500,100)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-4, 1e-3],
    'learning_rate_init': [0.001, 0.002]
}

# 初始化GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1)

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("model_file")
parser.add_argument("--feat_appendix", default=".csv")
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
    
    # For videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
        feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
        label_list.append(int(df_videos_label[video_id]))
y = np.array(label_list)
X = np.array(feat_list)

# 训练模型
grid_search.fit(X, y)

# 输出最佳参数和最佳得分
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)