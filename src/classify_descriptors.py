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

from matplotlib import pyplot as plt

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

    '''
    Things to work on:
    Vary up the dataset:
        - Classify the total image instead of just one keypoint
            - Learn based on the classification of all of the keypoints in the 
            image and their location
        - With classification of image in hand, classify image with random
        perturbations
            - rotate the image gaussian ammount
            - add gaussian noise to the whole image
            - add gaussian brightness to the whole image
            - add guassian darkness
            - red
            - green
            - blue
            - shift left, right, up, down (should be no biggy, can skip because 
            keypoints will just shift)
            - image flip left/right, up/down?
            - scaling image (zoom center)
            - affine transform
            - perspective transform
    Once I've played with varying the dataset, I should either find a set of 
    images that confirm the stopsign is pretty robust or an expanded training
    set to train with. From there, optimize KNN, 
    '''

    # load data from csv, split into training and test sets
    print('begin loading data')
    train_X, train_y, test_X, test_y = load_data(12345)

    Klassifiers = [
        # GradientBoostingClassifier, 
        # GaussianProcessClassifier, # This gave a MemoryError on round 0/6
        # SGDClassifier, 
        KNeighborsClassifier, 
        # MLPClassifier, SVC,
        # DecisionTreeClassifier
        ]
    num_tests = 10
    for index, Klassifier in enumerate(Klassifiers):
        acc = []
        pre = []
        rec = []
        for num_neighbors in range(0, 7):
            print('num neighbors %d' % (num_neighbors + 1,))
            acc_accum = 0
            pre_accum = 0
            rec_accum = 0
            for seed in range(0, num_tests):
                print('round %4d/%4d' % (seed, num_tests))
                train_X, train_y = subsample_data(train_X, train_y, 0.5, seed+9001)
                # print('begin fitting')
                classifier = Klassifier(n_neighbors=num_neighbors+1)
                classifier.fit(train_X, train_y)
                # print('end fitting')

                # print('begin pred')
                y_pred = classifier.predict(test_X)
                # print('end pred')
                # print('begin scoring')
                acc_accum += accuracy_score(y_true=test_y, y_pred=y_pred)
                pre_accum += precision_score(y_true=test_y, y_pred=y_pred)
                rec_accum += recall_score(y_true=test_y, y_pred=y_pred)
                # print('end scoring')
            acc.append(acc_accum / num_tests)
            pre.append(pre_accum / num_tests)
            rec.append(rec_accum / num_tests)
        print(Klassifier)
        print('a: %.4f (percent correctly classified)' % (sum(acc)/len(acc),))
        print('p: %.4f (percent of correct positives)' % (sum(pre)/len(pre),))
        print('r: %.4f (percent of positive results found)' % (sum(rec)/len(rec),))
        print('---')
        print('plot data')
        print(pre)
        print(rec)
        plt.plot(x=pre, y=rec) # , c=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.show()
