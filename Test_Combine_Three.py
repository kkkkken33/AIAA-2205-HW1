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
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
args = parser.parse_args()

# Load models
mlp_model = pickle.load(open(r"./models/mfcc-66.mlp.model", "rb"))
svm_model = pickle.load(open(r"./models/mfcc-66.svm.multiclass.model", "rb"))
stack_model = pickle.load(open(r"./models/stack_model.66.model", "rb"))
# Create array containing features of each sample
fread = open(args.list_videos, "r")
feat_list = []
video_ids = []
for line in fread.readlines():
# HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
        feat_list.append(np.zeros(args.feat_dim))
    else:
        feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

X = np.array(feat_list)

# Get predictions
mlp_pred = mlp_model.predict_proba(X)
svm_pred = svm_model.predict_proba(X)
combined_pred = np.hstack((mlp_pred, svm_pred))
final_predictions = stack_model.predict(combined_pred)
with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(final_predictions):
        f.writelines("%s,%d\n" % (video_ids[i], pred_class))