#!/bin/python

import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import argparse
from sklearn.preprocessing import MinMaxScaler

# Train AdaBoost classifier with labels

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


    # Train AdaBoost model
    base_estimator = DecisionTreeClassifier(max_depth=10)  # Shallow tree for boosting
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,  # Number of weak learners
        learning_rate=1.0,  # Contribution of each classifier
        random_state=42
    )
    model.fit(X, y)

    # save trained AdaBoost model in output_file
    pickle.dump(model, open(args.output_file, 'wb'))
    print('AdaBoost classifier trained successfully')
