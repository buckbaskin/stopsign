#!/usr/bin/env python

import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

pkg_path = '/home/buck/ros_ws/src/stopsign'
BULK_DATA_FILE = '%s/data/013_extra_man_labels/clean_100.csv' % (pkg_path,)
DTC_MODEL_STORE_FILE = '%s/data/013_extra_man_labels/dtc_classifier01.pkl' % (pkg_path,)

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

if __name__ == '__main__':
    print('save classifier trained on all data')
    classifier = DecisionTreeClassifier(max_depth=15)
    bigX, bigy = load_data()
    print('big classification')
    classifier.fit(bigX, bigy)
    print('save classifier')
    joblib.dump(classifier, DTC_MODEL_STORE_FILE)
