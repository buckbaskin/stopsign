#!/usr/bin/env python
import rospkg

import cv2
import datetime
import numpy as np
import pandas as pd
import joblib
import platform

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

KLASSIFIER_PATH = '%s/data/017_the_500/competition_classifier01_%s.pkl' % (pkg_path, platform.python_version(),)
REDUCER_PATH = '%s/data/017_the_500/competition_reducer01_%s.pkl' % (pkg_path, platform.python_version(),)

IMAGE_BASE_STRING = '%s/data/011_new_tests/%s' % (pkg_path, '%02d/frame%04d.jpg',)
OUT_BASE_STRING = '%s/data/018_demo_video/%s' % (pkg_path, 'try%02d/frame%04d.jpg',)

start_image_id = 1
end_image_id = 1093

def get_image(video_id, image_id):
    filename = IMAGE_BASE_STRING % (video_id, image_id,)
    # print('filename %s' % (filename,))
    return cv2.imread(filename, cv2.IMREAD_COLOR)

num_features = 500
preset = np.zeros((500, 256,))

def colorize_image(img, classifier, reducer):
    height, width, channels = img.shape
    area = height*width
    orb = cv2.ORB_create(nfeatures = num_features)
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    global preset
    preset = np.unpackbits(des, axis=1)
    # classify descriptors
    print('begin predict')
    start_time = datetime.datetime.now()

    # TODO transform and expand the data from int to bitwise

    smol_des = reducer.transform(preset)
    classes = classifier.predict(smol_des)
    # show how long the classification takes
    print('end predict. %.5f sec' % ((datetime.datetime.now() - start_time).total_seconds()))

    # rebuild list of keypoints that classify as a stopsign
    in_kp = []
    out_kp = []
    for index, kp in enumerate(kp):
        if classes[index] == 1:
            in_kp.append(kp)
        else:
            out_kp.append(kp)

    # print('short kp')
    # print(len(in_kp))
    # for index, kp in enumerate(in_kp):
    #     print('% 4d x: % 4d y: % 4d' % (index, kp.pt[0], kp.pt[1]))

    top_left = (0,0)
    bottom_right = (img.shape[1], img.shape[0])
    # if the list of keypoints is longer than 3, make edge red, draw kp

    img = cv2.drawKeypoints(
        image=img,
        keypoints=in_kp,
        outImage=img,
        color=(100,255,0),
        flags=int(cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    img = cv2.drawKeypoints(
        image=img,
        keypoints=out_kp,
        outImage=img,
        color=(0,0,255),
        flags=int(cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG))
    # cv2.imshow('preview', img)
    # val = cv2.waitKey(0) % 256

    if len(in_kp) > 1:
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


def set_image(img, video_id, image_id):
    filename = OUT_BASE_STRING % (video_id, image_id,)
    cv2.imwrite(filename, img)

if __name__ == '__main__':
    # load data from csv, split into training and test sets
    classifier = joblib.load(KLASSIFIER_PATH)  
    reducer = joblib.load(REDUCER_PATH)
    for video_id in range(3, 4):
        for image_id in range(start_image_id, 101): # end_image_id):
            if image_id % 1 == 0:
                print('%02d %d / %d' % (video_id, image_id, end_image_id,))
            og_img = get_image(video_id, image_id)
            if og_img is None:
                continue
            noise_img = colorize_image(og_img, classifier, reducer)
            set_image(noise_img, video_id, image_id)
