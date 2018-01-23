#!/usr/bin/env python3

import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor

pkg_path = '/home/buck/ros_ws/src/stopsign'

IMAGE_RATE = 30 # hz

POSITIVE_BITS_FILE = '%s/data/017_the_500/positive_bits_500.csv' % (pkg_path,)
NEGATIVE_BITS_FILE = '%s/data/017_the_500/negative_bits_500%s.csv' % (pkg_path, '_%d')

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
        MLPClassifier, 
    ]

    # experimentally, about 300-320 nodes is the max
    mlp_spec = {
        'activation': ['tanh',],
        'solver': ['adam',], # ['lbfgs', 'sgd', 'adam',],
        'alpha': [10**-1,],
        # accuracy was 75%+ at 80 for the first layer w/ 85% training acc
        # when trying to increase depth, the accuracy drops
        # so now, pushing width vs time tradeoff at 2 layers

        # 2 layers seems to be the most accurate with less overfitting
        'hidden_layer_sizes': [
            tuple([135,]*2),
        ],
        'verbose': [True,],
    }

    ensemble_spec = {
        'n_estimators': 12,
        'n_jobs': -1,
    }

    Klassifier_configs = []
    Klassifier_configs.extend(make_all_combinations(mlp_spec))

    Ensembler = BaggingRegressor

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
                    classifier = Ensembler(Klassifier(**config_setup), random_state=rng, **ensemble_spec)
                    # classifier = Klassifier(**config_setup)
                    print('begin fit')
                    start = datetime.datetime.now()
                    classifier.fit(train_X, train_y)
                    print('end fit | %.1f sec' % ((datetime.datetime.now() - start).total_seconds(),))
                    y_pred = classifier.predict(train_X)
                    y_pred = np.where(y_pred > 0.5, 1, 0)

                    train_acc_accum += accuracy_score(y_true=train_y, y_pred=y_pred)
                    
                    X_splits = np.array_split(test_X, split_count)
                    y_splits = np.array_split(test_y, split_count)

                    split_version = list(zip(X_splits, y_splits))
                    count = 0
                    for test_X_sub, test_y_sub in split_version:
                        count += 1
                        print('sub round %4d/%4d | %.1f sec' % (count, len(split_version), (datetime.datetime.now() - start).total_seconds(),))
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
                avg_time = tim_accum / (num_tests*split_count)
                print('t: %.6f sec' % (avg_time,))
                if avg_time > (1.0 / 30):
                    print('stopping based on hard time limit')
                    break

            print(Klassifier)
            print(Ensembler)
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
