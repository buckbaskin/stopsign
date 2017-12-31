#!/usr/bin/env python3

# import cv2
import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# import matplotlib
# matplotlib.use('TKAgg')
# from matplotlib import pyplot as plt

# class  ,descr00,descr01,descr02,descr03,descr04,descr05,descr06,descr07,
# descr08,descr09,descr10,descr11,descr12,descr13,descr14,descr15,descr16,
# descr17,descr18,descr19,descr20,descr21,descr22,descr23,descr24,descr25,
# descr26,descr27,descr28,descr29,descr30,descr31,angle  ,classid,octave ,
# x      ,y      ,respons,size   ,imageid

pkg_path = '/home/buck/ros_ws/src/stopsign'

IMAGE_RATE = 30 # hz

POSITIVE_BITS_FILE = '%s/data/016_bit_classifiers/positive_bits_200.csv' % (pkg_path,)
NEGATIVE_BITS_FILE = '%s/data/016_bit_classifiers/negative_bits_200%s.csv' % (pkg_path, '_%d')

bit_label = 'd%02db%01d'

descriptors = []
for i in range(32):
    for bit_index in range(0, 8):
        descriptors.append('d%02db%01d' % (i, bit_index,))

klass = ['class']

def load_data(pos_data_file, neg_data_file):
    pdf = pd.read_csv(pos_data_file, header=0)
    ndf = pd.read_csv(neg_data_file, header=0)
    df = pd.concat([pdf, ndf,]).sample(frac=1.0).reset_index(drop=True)

    # split into class, features
    X = df[descriptors]
    y = df[klass]
    print('X.describe()')
    print(X.describe())
    print('y.describe()')
    print(y.describe())
    return X, y

def scramble_data(X, y, seed=None):

    # use mask to split into test, train
    if seed is not None:
        np.random.seed(seed)
    msk = np.random.rand(len(X)) < 0.7
    
    train_X = X[msk].as_matrix()
    test_X = X[~msk].as_matrix()
    train_y = y[msk].as_matrix().ravel()
    test_y = y[~msk].as_matrix().ravel()
    return train_X, train_y, test_X, test_y

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

    Klassifiers = [
        DecisionTreeRegressor, 
    ]

    dtr_spec = {
        'n_estimators': list(range(1,302, 30)),
    }

    Klassifier_configs = []
    Klassifier_configs.extend(make_all_combinations(dtr_spec))

    bigX, bigy = load_data(POSITIVE_BITS_FILE, NEGATIVE_BITS_FILE % 0)

    split_count = 8

    for num_tests in range(8,9):
        for index, Klassifier in enumerate(Klassifiers):
            acc = []
            train_acc = []
            pre = []
            rec = []
            tim = []

            for config_setup in Klassifier_configs:
                print('current config: %s' % (config_setup,))
                acc_accum = 0
                train_acc_accum = 0
                pre_accum = 0
                rec_accum = 0
                tim_accum = 0
                for seed in range(0, num_tests):
                    print('round %4d/%4d' % (seed+1, num_tests))
                    train_X, train_y, test_X, test_y = scramble_data(bigX, bigy, seed)
                    
                    rng = np.random.RandomState(seed+1)
                    classifier = AdaBoostRegressor(Klassifier(max_depth=7), random_state=rng, **config_setup)
                    classifier.fit(train_X, train_y)
                    y_pred = classifier.predict(train_X)
                    y_pred = np.where(y_pred > 0.5, 1, 0)

                    train_acc_accum += accuracy_score(y_true=train_y, y_pred=y_pred)
                    
                    X_splits = np.array_split(test_X, split_count)
                    y_splits = np.array_split(test_y, split_count)

                    split_version = zip(X_splits, y_splits)
                    for test_X_sub, test_y_sub in split_version:
                        stime = datetime.datetime.now()
                        y_pred = classifier.predict(test_X_sub)
                        y_pred = np.where(y_pred > 0.5, 1, 0)
                        etime = datetime.datetime.now()

                        acc_accum += accuracy_score(y_true=test_y_sub, y_pred=y_pred)
                        pre_accum += precision_score(y_true=test_y_sub, y_pred=y_pred)
                        rec_accum += recall_score(y_true=test_y_sub, y_pred=y_pred)
                        tim_accum += (etime - stime).total_seconds()

                train_acc.append(train_acc_accum / (num_tests))
                acc.append(acc_accum / (num_tests*split_count))
                pre.append(pre_accum / (num_tests*split_count))
                rec.append(rec_accum / (num_tests*split_count))
                tim.append(tim_accum / (num_tests*split_count))
                print('a: %.4f (percent correctly classified)' % (acc_accum / (num_tests*split_count),))
                print('ta:%.4f (percent correctly classified in training)' % (train_acc_accum / num_tests,))
                print('p: %.4f (percent of correct positives)' % (pre_accum / (num_tests*split_count),))
                print('r: %.4f (percent of positive results found)' % (rec_accum / (num_tests*split_count),))
                print('t: %.6f sec' % (tim_accum / (num_tests*split_count),))       

            print(Klassifier)
            print('Averaged over %d tests' % (num_tests,))
            # better accuracy summary
            print('a: %.4f (avg percent correctly classified)' % (sum(acc)/len(acc),))
            print('Top Accuracies')
            print('90 percent of max accuracy cutoff')
            sorted_ = list(sorted(enumerate(acc), key=lambda x: -x[1]))
            top_acc = sorted_[0][1]
            sorted_ = list(filter(lambda x: x[1] >= top_acc * 0.9, sorted_))
            for acc_index, accuracy in sorted_[:15]:
                print('% 4.2f | %s' % (accuracy * 100, Klassifier_configs[acc_index],))
                print('% 4.2f | training accuracy' % (train_acc[acc_index] * 100.0,))

            print('p: %.4f (avg percent of correct positives)' % (sum(pre)/len(pre),))
            print('r: %.4f (avg percent of positive results found)' % (sum(rec)/len(rec),))

            print('t: %.6f avg sec' % (sum(tim) / len(tim)))
            print('Top Prediction Latencies')
            print('Top 10')
            sorted_ = list(sorted(enumerate(tim), key=lambda x: x[1]))
            for tim_index, pred_latency in sorted_[:10]:
                print('%.6f | %s' % (pred_latency, Klassifier_configs[tim_index]))
