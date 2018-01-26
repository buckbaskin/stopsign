#!/usr/bin/env python
import datetime
import joblib
import numpy as np
import pandas as pd
import rospkg

from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

BULK_DATA_FILE = '%s/data/003_manual_labels/all.csv' % (pkg_path,)

KP_MODEL_STORE_FILE = '%s/data/005_image_labels/kp_classifier.pkl' % (pkg_path,)

start_image_id = 0
end_image_id = 2189

NUM_IMAGES = end_image_id - start_image_id

descriptors = []
for i in range(32):
    descriptors.append('descr%02d' % (i,))

klass = ['class'.ljust(7), 'imageid']

def load_data(seed=None):
    df = pd.read_csv(BULK_DATA_FILE, header=0)
    # mutate data back from stored form
    df['class  '] = df['class  '].apply(lambda cls: cls / 1000.0)
    df['angle  '] = df['angle  '].apply(lambda ang: ang / 1000.0)
    df['respons'] = df['respons'].apply(lambda res: res / 100000000.0)

    # split into class, features
    X = df[descriptors]
    y = df[klass]
    # print(y.describe())

    # use mask to split into test, train
    if seed is not None:
        np.random.seed(seed)
    img_msk = np.random.rand(NUM_IMAGES) < 0.8
    X['train'] = y['imageid'].apply(lambda x: img_msk[x])
    X_msk = X['train'] == 1
    y['train'] = y['imageid'].apply(lambda x: img_msk[x])
    y_msk = y['train'] == 1

    X.drop('train', axis=1, inplace=True)
    train_X = X[X_msk].as_matrix()
    train_y = y['class  '][y_msk].as_matrix().ravel()
    return train_X, train_y

def train_and_save_kp_classifier():
    start_time = datetime.datetime.now()
    print('Time 0 sec')
    print('begin loading data')
    train_X, train_y = load_data(seed=12345)

    print('train kp classifier %.2f sec' % ((datetime.datetime.now() - start_time).total_seconds(),))
    kp_nbrs = KNeighborsClassifier(n_jobs=1)
    kp_nbrs.fit(train_X, train_y)

    print('and now, predict %.2f sec' % ((datetime.datetime.now() - start_time).total_seconds(),))

    print('kp classifier %.2f sec' % ((datetime.datetime.now() - start_time).total_seconds(),))
    print('a:')
    
    print('save trained classifier %.2f sec' % ((datetime.datetime.now() - start_time).total_seconds(),))
    joblib.dump(kp_nbrs, KP_MODEL_STORE_FILE)

    print('saved classifier %.2f sec' % ((datetime.datetime.now() - start_time).total_seconds(),))

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

    print('collect after classifier')