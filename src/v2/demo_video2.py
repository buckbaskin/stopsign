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

GREY_STOPSIGN = '%s/data/018_demo_video/stop_sign_grey.jpg' % (pkg_path,)

start_image_id = 1
end_image_id = 1093

def get_image(video_id, image_id):
    filename = IMAGE_BASE_STRING % (video_id, image_id,)
    print('filename %s' % (filename,))
    return cv2.imread(filename, cv2.IMREAD_COLOR)

num_features = 5000
preset = np.zeros((500, 256,))

SSG = cv2.imread(GREY_STOPSIGN, cv2.IMREAD_COLOR)
if SSG is None:
    print('Image not loaded')
    import sys
    sys.exit(1)
ssgorb = cv2.ORB(nfeatures = 500)
ssgorb = cv2.ORB_create(nfeatures = 100)
ssgkp = ssgorb.detect(SSG, None)
ssgkp, ssgdes = ssgorb.compute(SSG, ssgkp)

buckfm = cv2.BFMatcher(cv2.NORM_HAMMING)

orb = cv2.ORB(nfeatures = num_features)
orb = cv2.ORB_create(nfeatures = num_features)

def classify_image(image, classifier, reducer):
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    # kp to bitwise numpy array
    global preset
    preset = np.unpackbits(des, axis=1)
    
    X = preset
    smol_X = reducer.transform(X)
    y = classifier.predict(smol_X)

    matches = buckfm.match(ssgdes, des)
    matches.sort(key= lambda match: match.distance)
    matches = list(filter(lambda match: match.distance < 50, matches))
    outImg = np.zeros((1000,1000,3), np.uint8)
    outImg = cv2.drawMatches(SSG, ssgkp, image, kp, matches, outImg=outImg, flags=0)
    cv2.imshow('matching!', outImg)
    cv2.waitKey(200)
    # classify image based on match count
    if np.sum(y) > 10:
        # publish true on stopsign channel
        # pub_buddy.publish(Bool(True))
        return True
    else:
        # publish false on stopsign channel
        # pub_buddy.publish(Bool(False))
        return False

def colorize_image(img, classifier, reducer):
    # else make edge green
    has_stopsign = classify_image(img, classifier, reducer)

    top_left = (0,0)
    bottom_right = (img.shape[1], img.shape[0])
    # if the list of keypoints is longer than 3, make edge red, draw kp

    # img = cv2.drawKeypoints(
    #     image=img,
    #     keypoints=in_kp,
    #     outImage=img,
    #     color=(100,255,0),
    #     flags=int(cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    # img = cv2.drawKeypoints(
    #     image=img,
    #     keypoints=out_kp,
    #     outImage=img,
    #     color=(0,0,255),
    #     flags=int(cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG))
    # cv2.imshow('preview', img)
    # val = cv2.waitKey(0) % 256

    if has_stopsign:
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
    print('loaded')
    for video_id in range(01, 25):
        print('video')
        for image_id in range(start_image_id, end_image_id): # end_image_id):
            if image_id % 1 == 0:
                print('%02d %d / %d' % (video_id, image_id, end_image_id,))
            og_img = get_image(video_id, image_id)
            if og_img is None:
                print('og_img is None')
                continue
            print('colorize image')
            noise_img = colorize_image(og_img, classifier, reducer)
            set_image(noise_img, video_id, image_id)
