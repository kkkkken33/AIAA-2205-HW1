import argparse
import numpy as np
import os
import pickle
import numpy as np
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

args = parser.parse_args()

model = pickle.load(open(args.model_file, "rb"))

fread = open(args.list_videos, "r")
feat_list = []
video_ids = []
for line in fread.readlines():
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
        feat_list.append(np.zeros(args.feat_dim))
    else:
        feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

X = np.array(feat_list)

pred_classes = model.predict(X)

with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
        f.writelines("%s,%d\n" % (video_ids[i], pred_class))
