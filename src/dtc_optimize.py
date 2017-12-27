#!/usr/bin/env python
import rospkg

import cv2
import datetime
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

# class  ,descr00,descr01,descr02,descr03,descr04,descr05,descr06,descr07,
# descr08,descr09,descr10,descr11,descr12,descr13,descr14,descr15,descr16,
# descr17,descr18,descr19,descr20,descr21,descr22,descr23,descr24,descr25,
# descr26,descr27,descr28,descr29,descr30,descr31,angle  ,classid,octave ,
# x      ,y      ,respons,size   ,imageid

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

IMAGE_RATE = 11 # hz

BULK_DATA_FILE = '%s/data/003_manual_labels/all.csv' % (pkg_path,)

start_image_id = 0
end_image_id = 2189

SAVE_IMAGE_FILE = '%s/data/009_dtc_opt/%s' % (pkg_path, 'depth_%03dtest_entropy3.png')

descriptors = []
for i in range(32):
    descriptors.append('descr%02d' % (i,))

klass = ['class'.ljust(7)]

def load_data(seed=None):
    df = pd.read_csv(BULK_DATA_FILE, header=0)
    # mutate data back from stored form
    df['class  '] = df['class  '].apply(lambda cls: cls / 1000.0)
    df['angle  '] = df['angle  '].apply(lambda ang: ang / 1000.0)
    df['respons'] = df['respons'].apply(lambda res: res / 100000000.0)

    # split into class, features
    X = df[descriptors]
    y = df[klass]
    print('X.describe()')
    print(X.describe())
    print('y.describe()')
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

def increment_index_list(index, max_list):
    index[-1] += 1
    if index[-1] >= max_list[-1]:
        for i in range(len(index) - 1, 0, -1):
            if index[i] >= max_list[i]:
                index[i] = 0
                index[i-1] += 1

    return index

def make_all_combinations(dict_of_arglists):
    input_args = list(dict_of_arglists.keys())
    max_list = []
    for input_key in input_args:
        max_list.append(len(dict_of_arglists[input_key]))
    index_list = [0] * len(input_args)

    count = 1
    for val in max_list:
        count *= val

    for _ in range(count):
        input_vals = []
        for index, input_key in enumerate(input_args):
            input_vals.append(dict_of_arglists[input_key][index_list[index]])
        combined = zip(input_args, input_vals)
        d = dict(combined)
        yield d
        index_list = increment_index_list(index_list, max_list)

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
    train_X, train_y, test_X, test_y = load_data(1234567)

    Klassifiers = [
        DecisionTreeClassifier, 
    ]

    max_iters = list(range(1,30))
    sgd_spec = {
        'criterion': ['entropy',], # ['gini', 'entropy',],
        'max_depth': max_iters,
    }

    Klassifier_configs = []
    Klassifier_configs.extend(make_all_combinations(sgd_spec))

    num_tests = 8
    for i in range(3,4):
        num_tests = 2**i

        for index, Klassifier in enumerate(Klassifiers):
            acc = []
            pre = []
            rec = []
            tim = []

            for config_setup in Klassifier_configs:
                print('current config: %s' % (config_setup,))
                acc_accum = 0
                pre_accum = 0
                rec_accum = 0
                tim_accum = 0
                for seed in range(0, num_tests):
                    # print('round %4d/%4d' % (seed+1, num_tests))
                    train_X, train_y = subsample_data(train_X, train_y, 0.5, seed*num_tests+9105)
                    
                    classifier = Klassifier(**config_setup)
                    classifier.fit(train_X, train_y)
                    
                    X_splits = np.array_split(test_X, 437)
                    y_splits = np.array_split(test_y, 437)

                    split_version = zip(X_splits, y_splits)
                    for test_X_sub, test_y_sub in split_version:
                        # print('begin pred')
                        stime = datetime.datetime.now()
                        y_pred = classifier.predict(test_X_sub)
                        etime = datetime.datetime.now()
                        # print('end pred')
                        # print('begin scoring')
                        acc_accum += accuracy_score(y_true=test_y_sub, y_pred=y_pred)
                        pre_accum += precision_score(y_true=test_y_sub, y_pred=y_pred)
                        rec_accum += recall_score(y_true=test_y_sub, y_pred=y_pred)
                        tim_accum += (etime - stime).total_seconds()
                    # print('end scoring')
                acc.append(acc_accum / (num_tests*437))
                pre.append(pre_accum / (num_tests*437))
                rec.append(rec_accum / (num_tests*437))
                tim.append(tim_accum / (num_tests*437))
                print('a: %.4f (percent correctly classified)' % (acc_accum / (num_tests*437),))
                print('p: %.4f (percent of correct positives)' % (pre_accum / (num_tests*437),))
                print('r: %.4f (percent of positive results found)' % (rec_accum / (num_tests*437),))
                print('t: %.6f sec' % (tim_accum / (num_tests*437),))       

            print(Klassifier)
            print('Averaged over %d tests' % (num_tests,))
            # better accuracy summary
            print('a: %.4f (avg percent correctly classified)' % (sum(acc)/len(acc),))
            print('Top Accuracies')
            print('90 percent of max accuracy cutoff')
            sorted_ = sorted(enumerate(acc), key=lambda x: -x[1])
            top_acc = sorted_[0][1]
            sorted_ = filter(lambda x: x[1] >= top_acc * 0.9, sorted_)
            for acc_index, accuracy in sorted_[:15]:
                print('% 4.2f | %s' % (accuracy * 100, Klassifier_configs[acc_index]))

            print('p: %.4f (avg percent of correct positives)' % (sum(pre)/len(pre),))
            print('r: %.4f (avg percent of positive results found)' % (sum(rec)/len(rec),))

            print('t: %.6f avg sec' % (sum(tim) / len(tim)))
            print('Top Prediction Latencies')
            print('Top 10')
            sorted_ = sorted(enumerate(tim), key=lambda x: x[1])
            for tim_index, pred_latency in sorted_[:10]:
                print('%.6f | %s' % (pred_latency, Klassifier_configs[tim_index]))

            print(num_tests)
            print(max_iters)
            print(acc)
            plt.plot(max_iters, acc, label=str(num_tests))
            plt.axis([min(max_iters) - 1, max(max_iters) + 1, 0, 1.0])
            plt.xlabel('Max Depth')
            plt.ylabel('Accuracy')
            plt.savefig(SAVE_IMAGE_FILE % (num_tests,))
            # plt.show()
            plt.clf()
            print('did it save?')