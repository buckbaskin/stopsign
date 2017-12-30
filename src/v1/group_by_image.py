#!/usr/bin/env python
import rospkg

import cv2
import datetime
import gc
import joblib
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

BULK_DATA_FILE = '%s/data/003_manual_labels/all.csv' % (pkg_path,)

IMAGE_DATA_FILE = '%s/data/005_image_labels/all.csv' % (pkg_path,)

KP_MODEL_STORE_FILE = '%s/data/005_image_labels/kp_classifier.pkl' % (pkg_path,)

start_image_id = 0
end_image_id = 2189

NUM_IMAGES = end_image_id - start_image_id
NUM_ORBS_FEATURES = 500

labels = ['class']
for i in range(0, 32):
    labels.append('descr%02d' % (i,))
labels.extend(['angle', 'classid', 'octave', 'x', 'y', 'response', 'size','imageid'])


new_labels = ['class']
for i in range(0, 500):
    labels.append('orbkp%03d' % i)

def transfer(read_file, write_file):
    print('time 0 sec')
    start_time = datetime.datetime.now()
    kp_classifier = joblib.load(KP_MODEL_STORE_FILE)
    write_file.write(','.join(labels) + '\n')

    first_line = True
    imageid = 0
    kp_id = 0
    image_has_stopsign = False
    writeline = [None] * 501

    output_batch = []

    for line in read_file:
        if first_line:
            first_line = False
            continue

        values = [float(x.strip()) for x in line.split(',')]
        values[labels.index('class')] /= 1000.0
        values[labels.index('angle')] /= 1000.0
        values[labels.index('response')] /= 100000000.0

        read_image_id = values[labels.index('imageid')]
        # print('read %d vs imgid %d' % (read_image_id, imageid,))
        if read_image_id == imageid:
            X = np.array(values[1:33]).reshape(1, -1)
            y_pred = kp_classifier.predict(X)[0]
            image_has_stopsign = image_has_stopsign or y_pred > 0.5
            writeline[kp_id + 1] = '%.3f' % float(y_pred)
            if kp_id >= 499:
                kp_id = 498
            kp_id += 1
        elif read_image_id - imageid == 1:
            assert kp_id == 499
            img_class = image_has_stopsign
            writeline[0] = str(int(img_class))

            output_batch.append(','.join(writeline) + '\n')
            if imageid % 20 == 0:
                print('Batching image %4d / %4d @ %.2f sec total %.2f sec per' % (
                    imageid + 1,
                    end_image_id,
                    (datetime.datetime.now() - start_time).total_seconds(),
                    (datetime.datetime.now() - start_time).total_seconds() / (imageid+1),))

            imageid += 1
            kp_id = 0
            image_has_stopsign = False
        else:
            raise ValueError('Unexpected value for imageid %d from %d' % (read_image_id, imageid))
        if len(output_batch) > 100:
            write_file.write(''.join(output_batch))
            print('write batch 100 %.2f sec total %.2f sec per' % (
                (datetime.datetime.now() - start_time).total_seconds(),
                (datetime.datetime.now() - start_time).total_seconds() / (imageid+1)))
            output_batch = []

if __name__ == '__main__':
    ### Begin the whole process ###

    with open(BULK_DATA_FILE, 'r') as kp_f:
        with open(IMAGE_DATA_FILE, 'w') as im_f:
            transfer(kp_f, im_f)