#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys
from sklearn.preprocessing import MinMaxScaler

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  scaler = MinMaxScaler()
  X = scaler.fit_transform(X)


  # TA: write your code here 
  model = MLPClassifier(
        hidden_layer_sizes=(600, ),  # 一层包含100个神经元
        activation='relu',           # 激活函数
        solver='adam',               # 优化算法
        alpha=0.0001,                # L2惩罚参数
        batch_size='auto',           # 批量大小
        learning_rate='constant',     # 学习率策略
        learning_rate_init=0.001,    # 初始学习率
        max_iter=200,                # 最大迭代次数
        random_state=42              # 随机种子
    )
  model.fit(X, y)
  
  # save trained MLP in output_file
  pickle.dump(model, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
