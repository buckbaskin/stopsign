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
from sklearn.naive_bayes import GaussianNB
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
    msk = np.random.rand(len(df)) < 0.8
    train_X = X[msk].as_matrix()
    test_X = X[~msk].as_matrix()
    train_y = y[msk][:,0].as_matrix().ravel()
    test_y = y[~msk][:,0].as_matrix().ravel()
    train_id = y[msk][:,1].as_matrix().ravel()
    test_id = y[~msk][:,1].as_matrix().ravel()
    return train_X, train_y, train_id, test_X, test_y, test_id

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

    '''
    Things to work on:
    Vary up the dataset:
        - Classify the total image instead of just one keypoint
            - Learn based on the classification of all of the keypoints in the 
            image and their location
    '''

    # load data from csv, split into training and test sets
    print('begin loading data')
    train_X, train_y, train_id, test_X, test_y, test_id = load_data(12345)

    kp_nbrs = KNeighborsClassifier()
    # train_X, train_y = subsample_data(train_X, train_y, ratio=0.5, seed=123456)
    kp_nbrs.fit(train_X, train_y)

    # I need to group by images first

    NUM_IMAGES = end_image_id - start_image_id
    NUM_ORBS_FEATURES = 500

    image_train_X = np.zeros(NUM_IMAGES, NUM_ORBS_FEATURES)
    image_train_y = np.zeros(NUM_IMAGES).ravel()

    # compile images into features (classification of keypoints)
    #   and the actual label for the image
    for i in range(start_image_id, end_image_id):
        mask = train_id == i
        subset_X = train_X[mask]
        subset_y = train_y[mask]

        predict_y = kp_nbrs.predict(subset_X).flatten()
        image_train_X[i, :] = predict_y
        image_train_y = (train_y == 1).any()

    gnb = GaussianNB()
    gnb.fit(image_train_X, image_train_y)