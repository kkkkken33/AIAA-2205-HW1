import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import os
import pickle

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

model = xgb.XGBClassifier(n_estimators=300)
model.fit(X, y)

# save trained XGBoost model in output_file
pickle.dump(model, open(args.output_file, 'wb'))
