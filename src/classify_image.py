#!/usr/bin/env python
import rospkg

import cv2
import gc
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.external import joblib
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

IMAGE_RATE = 11 # hz

BULK_DATA_FILE = '%s/data/003_manual_labels/all.csv' % (pkg_path,)

IMAGE_DATA_FILE = '%s/data/004_image_labels/all.csv' % (pkg_path,)

KP_MODEL_STORE_FILE = '%s/data/004_image_labels/kp_classifier.pkl' % (pkg_path,)

start_image_id = 0
end_image_id = 2189

IMAGE_BASE_STRING = '%s/data/002_original_images/%s' % (pkg_path, 'frame%04d.jpg')

NUM_IMAGES = end_image_id - start_image_id
NUM_ORBS_FEATURES = 500

descriptors = []
for i in range(32):
    descriptors.append('descr%02d' % (i,))

klass = ['class'.ljust(7), 'imageid']

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
    img_msk = np.random.rand(NUM_IMAGES) < 0.8
    X['train'] = y['imageid'].apply(lambda x: img_msk[x])
    X_msk = X['train'] == 1
    y['train'] = y['imageid'].apply(lambda x: img_msk[x])
    y_msk = y['train'] == 1
    train_X = X[X_msk].as_matrix()
    test_X = X[~X_msk].as_matrix()
    train_y = y['class  '][y_msk].as_matrix().ravel()
    test_y = y['class  '][~y_msk].as_matrix().ravel()
    train_id = y['imageid'][y_msk].as_matrix().ravel()
    test_id = y['imageid'][~y_msk].as_matrix().ravel()
    return train_X, train_y, train_id, test_X, test_y, test_id

def load_data_by_image(start_id, end_id, seed=12345):
    # lazy load image data?
    df = pd.read_csv(BULK_DATA_FILE, header=0, skiprows=lambda x: 1 <= x <= start_id*500, nrows=500*(end_id - start_id))
    print(df.describe())

    gby_imageid = df.groupby(['imageid'])

    for name, indices in gby_imageid:
        subdf = df.iloc[indices]
    # split into class, features
        X = subdf[descriptors]
        y = subdf[klass]
        yield X, y

def subsample_data(X, y, ratio=0.5, seed=None):
    size = 1100
    rus = RandomUnderSampler(
        ratio={
            0: int(size * ratio),
            1: int(size * (1 - ratio)),
        },
        random_state=seed)
    return rus.fit_sample(X, y)

def train_and_save_kp_classifier():
    print('begin loading data')
    train_X, train_y, train_id, test_X, test_y, test_id = load_data(seed=12345)

    print('train kp classifier')
    kp_nbrs = KNeighborsClassifier()
    # train_X, train_y = subsample_data(train_X, train_y, ratio=0.5, seed=123456)
    kp_nbrs.fit(train_X, train_y)

    print('save trained classifier')
    joblib.dump(kp_nbrs, KP_MODEL_STORE_FILE)

def load_kp_and_write_data_by_image():
    kp_nbrs = joblib.load(KP_MODEL_STORE_FILE)
    step = 2188 // 4
    index = 0

    def columns():
        yield 'class'
        yield 'imageid'
        for i in range(0, NUM_ORBS_FEATURES):
            yield ('orbkp%03d' % (i,))

    out = pd.DataFrame(columns = columns())
    outline = np.zeroes((1, NUM_ORBS_FEATURES + 2))
    out.to_csv(IMAGE_DATA_FILE)
    
    imageid = 0
    for i in range(0, 4):
        for X, y in load_data_by_image(i * step, i * step + step):
            # X, y grouped by image
            pred_y = kp_nbrs.predict(X)
            outline[0, 0] = (y == 1).any()
            outline[0, 0] = imageid
            outline[0, 2:] = pred_y.flatten()
            out.append(outline)
            imageid += 1

        with open(IMAGE_DATA_FILE, 'a') as f:
            out.to_csv(f, header=False)
        out.drop(out.index, inplace=True)

def load_image_labels_and_classify_images():
    df = pd.read_csv(IMAGE_DATA_FILE, header=0)
    # mutate data back from stored form
    def features():
        for i in range(0, NUM_ORBS_FEATURES):
            yield ('orbkp%03d' % (i,))
    # split into class, features
    X = df[features()]
    y = df['class']
    print(y.describe())

    np.random.seed(12345)
    msk = np.random.rand(len(df)) < 0.8
    train_X = X[msk].as_matrix()
    test_X = X[~msk].as_matrix()
    train_y = y[msk].as_matrix().ravel()
    test_y = y[~msk].as_matrix().ravel()
    
    train_X, train_y = subsample_data(train_X, train_y, ratio=0.5, seed=789)
    gnb = GaussianNB()
    gnb.fit(train_X, train_y)

    y_pred = gnb.predict(test_X, test_y)

    print(GaussianNB)
    print('a: %.4f (percent correctly classified)' % (accuracy_score(y_true=test_y, y_pred=y_pred),))
    print('p: %.4f (percent of correct positives)' % (precision_score(y_true=test_y, y_pred=y_pred),))
    print('r: %.4f (percent of positive results found)' % (recall_score(y_true=test_y, y_pred=y_pred),))
    print('---')


if __name__ == '__main__':
    ### Begin the whole process ###

    '''
    Things to work on:
    Vary up the dataset:
        - Classify the total image instead of just one keypoint
            - Learn based on the classification of all of the keypoints in the 
            image and their location
    '''
    train_and_save_kp_classifier()
    gc.collect()
    
    # load data from csv, split into training and test sets
    # classify all keypoints and write classifications back to disk
    
    load_kp_and_write_data_by_image()
    gc.collect()

    load_image_labels_and_classify_images()
    import sys
    sys.exit(1)
    gc.collect()
