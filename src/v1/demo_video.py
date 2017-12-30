#!/usr/bin/env python
import rospkg

import cv2
import datetime
import numpy as np
import pandas as pd
import joblib

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

KLASSIFIER_PATH = '%s/data/009_dtc_opt/dtc_classifier.pkl' % (pkg_path,)

IMAGE_BASE_STRING = '%s/data/002_original_images/%s' % (pkg_path, 'frame%04d.jpg',)
OUT_BASE_STRING = '%s/data/010_demo_images/%s' % (pkg_path, 'frame%04d.jpg',)

start_image_id = 0
end_image_id = 2189

def get_image(image_id):
    filename = IMAGE_BASE_STRING % (image_id,)
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def colorize_image(img, kp_classifier):
    orb = cv2.ORB()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # classify descriptors
    start_time = datetime.datetime.now()
    print('begin predict')
    classes = kp_classifier.predict(des)
    # show how long the classification takes
    print('end predict. %.5f sec' % ((datetime.datetime.now() - start_time).total_seconds()))

    # rebuild list of keypoints that classify as a stopsign
    short_kp = []
    for index, kp in enumerate(kp):
        if classes[index] == 1:
            short_kp.append(kp)

    print(len(short_kp))

    top_left = (0,0)
    bottom_right = (img.shape[1], img.shape[0])
    # if the list of keypoints is longer than 3, make edge red, draw kp
    if len(short_kp) > 1:
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)
        # img = cv2.drawKeypoints(
        #     img,
        #     filter(lambda x: cv2.pointPolygonTest(contour, x.pt, False) >= 0, kp),
        #     color=(0,255,0),
        #     flags=0)
    # else make edge green
    else:
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)

    return img


def set_image(img, image_id):
    filename = OUT_BASE_STRING % (image_id,)
    cv2.imwrite(filename, img)

if __name__ == '__main__':
    # load data from csv, split into training and test sets
    kp_classifier = joblib.load(KLASSIFIER_PATH)
    for image_id in range(start_image_id, end_image_id):
        if image_id % 1 == 0:
            print('%d / %d' % (image_id + 1, end_image_id + 1,))
        og_img = get_image(image_id)
        noise_img = colorize_image(og_img, kp_classifier)
        set_image(noise_img, image_id)
