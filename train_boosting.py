#!/bin/python

import numpy as np
import os
import pickle
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# Custom Gradient Boosting with MLP for classification

class CustomMLPGradientBoosting:
    def __init__(self, n_estimators=20, learning_rate=0.1, hidden_layer_sizes=(100,)):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.models = []

    def fit(self, X, y):
        # Initialize predictions
        self.predictions = np.zeros((X.shape[0], len(np.unique(y))))
        
        for _ in range(self.n_estimators):
            # Calculate the residuals (one-hot encoded)
            residuals = y - np.argmax(self.predictions, axis=1)
            residuals_one_hot = np.zeros_like(self.predictions)
            for i in range(len(residuals)):
                residuals_one_hot[i, residuals[i]] = 1

            # Fit an MLP to the residuals
            model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                  activation='relu', solver='adam',
                                  max_iter=200, random_state=42)
            model.fit(X, residuals_one_hot)
            self.models.append(model)
            # Update predictions
            self.predictions += self.learning_rate * model.predict_proba(X)

    def predict(self, X):
    # Initialize total predictions for each class
        total_predictions = np.zeros((X.shape[0], 10))
        
        for model in self.models:
            total_predictions += self.learning_rate * model.predict_proba(X)
        
        # Return the class with the highest probability
        return np.argmax(total_predictions, axis=1)

# Train custom gradient boosting with MLP for classification

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


    # Train custom MLP gradient boosting model for classification
    model = CustomMLPGradientBoosting(n_estimators=20, learning_rate=0.1, hidden_layer_sizes=(100,))
    model.fit(X, y)

    # Save trained model
    pickle.dump(model, open(args.output_file, 'wb'))
    print('Custom MLP Gradient Boosting classification model trained successfully')
