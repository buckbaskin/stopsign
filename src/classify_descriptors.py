#!/usr/bin/env python
import rospkg

import cv2
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

IMAGE_RATE = 11 # hz

BULK_DATA_FILE = '%s/data/003_manual_labels/all.csv' % (pkg_path,)

start_image_id = 0
end_image_id = 2189

IMAGE_BASE_STRING = '%s/data/002_original_images/%s' % (pkg_path, 'frame%04d.jpg')

descriptors = []
for i in range(32):
    descriptors.append('descr%02d' % (i,))

klass = ['class'.ljust(7)]

def get_image(image_id):
    filename = IMAGE_BASE_STRING % (image_id,)
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def load_data(seed=None):
    df = pd.read_csv(BULK_DATA_FILE, header=0)
    # mutate data back from stored form
    df['class  '] = df['class  '].apply(lambda cls: cls / 1000.0)
    df['angle  '] = df['angle  '].apply(lambda ang: ang / 1000.0)
    df['respons'] = df['respons'].apply(lambda res: res / 100000000.0)

    # split into class, features
    X = df[descriptors]
    y = df[klass]
    print(y.describe())

    # use mask to split into test, train
    if seed is not None:
        np.random.seed(seed)
    msk = np.random.rand(len(df)) < 0.8
    train_X = X[msk].as_matrix()
    test_X = X[~msk].as_matrix()
    train_y = y[msk].as_matrix().ravel()
    test_y = y[~msk].as_matrix().ravel()
    return train_X, train_y, test_X, test_y

def subsample_data(X, y, ratio=0.5, seed=None):
    size = 1100
    rus = RandomUnderSampler(
        ratio={
            0: int(size * ratio),
            1: int(size * (1 - ratio)),
        },
        random_state=seed)
    return rus.fit_sample(X, y)

if __name__ == '__main__':
    ### Begin the whole process ###

    # load data from csv, split into training and test sets
    print('begin loading data')
    train_X, train_y, test_X, test_y = load_data(12345)

    Klassifiers = [GradientBoostingClassifier, 
        # GaussianProcessClassifier, # This gave a MemoryError on round 0/6
        SGDClassifier, KNeighborsClassifier, MLPClassifier, SVC,
        DecisionTreeClassifier]
    num_tests = 6
    for index, Klassifier in enumerate(Klassifiers):
        acc = []
        pre = []
        rec = []
        for seed in range(0, num_tests):
            print('%s round %4d/%4d' % (Klassifier, seed, num_tests))
            train_X, train_y = subsample_data(train_X, train_y, 0.5, seed)
            # print('begin fitting')
            classifier = Klassifier()
            classifier.fit(train_X, train_y)
            # print('end fitting')

            # print('begin pred')
            y_pred = classifier.predict(test_X)
            # print('end pred')
            # print('begin scoring')
            acc.append(accuracy_score(y_true=test_y, y_pred=y_pred))
            pre.append(precision_score(y_true=test_y, y_pred=y_pred))
            rec.append(recall_score(y_true=test_y, y_pred=y_pred))
            # print('end scoring')
        print(Klassifier)
        print('a: %.4f (percent correctly classified)' % (sum(acc)/num_tests,))
        print('p: %.4f (percent of correct positives)' % (sum(pre)/num_tests,))
        print('r: %.4f (percent of positive results found)' % (sum(rec)/num_tests,))
        print('---')
