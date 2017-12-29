#!/usr/bin/env python

import joblib
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier

pkg_path = '/home/buck/ros_ws/src/stopsign'
BULK_DATA_FILE = '%s/data/013_extra_man_labels/clean_200.csv' % (pkg_path,)
DTC_MODEL_STORE_FILE = '%s/data/013_extra_man_labels/dtc_classifier06.pkl' % (pkg_path,)

descriptors = []
for i in range(32):
    descriptors.append('descr%02d' % (i,))

klass = ['class'.ljust(7)]

def load_data():
    df = pd.read_csv(BULK_DATA_FILE, header=0)
    # split into class, features
    X = df[descriptors]
    y = df[klass]
    print('X.describe()')
    print(X.describe())
    print('y.describe()')
    print(y.describe())
    return X, y

def subsample_data(X, y, ratio=0.5, seed=None):
    size = 1100
    rus = RandomUnderSampler(
        ratio={
            0: int(size * ratio),
            1: int(size * (1 - ratio)),
        },
        random_state=seed)
    return rus.fit_sample(X.as_matrix(), y.as_matrix().ravel())

if __name__ == '__main__':
    print('save classifier trained on all data')
    classifier = DecisionTreeClassifier(max_depth=15)
    bigX, bigy = load_data()
    print('subsample data')
    trainX, trainy = subsample_data(bigX, bigy, ratio=0.90)
    print('big classification')
    classifier.fit(trainX, trainy)
    print('save classifier')
    joblib.dump(classifier, DTC_MODEL_STORE_FILE)
