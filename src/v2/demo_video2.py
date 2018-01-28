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

IMAGE_BASE_STRING = '%s/data/019_stopsign_images/%s' % (pkg_path, 'frame%04d.jpg',)
OUT_BASE_STRING = '%s/data/020_demo_video/%s' % (pkg_path, 'frame%04d.jpg',)

start_image_id = 0
end_image_id = 1093

def get_image(video_id, image_id):
    filename = IMAGE_BASE_STRING % (image_id,)
    # print('filename %s' % (filename,))
    return cv2.imread(filename, cv2.IMREAD_COLOR)

GREY_STOPSIGN = '%s/data/019_stopsign_images/stopper%s.jpg' % (pkg_path, '%d')
NUM_FEATURES = 2000

SSG = []
ssgorb = []
ssgkp = []
ssgdes = []
for i in range(4):
    SSG.append(cv2.imread(GREY_STOPSIGN % i, cv2.IMREAD_COLOR))
    if SSG[-1] is None:
        print('Image not loaded %d %s' % (i, GREY_STOPSIGN % i,))
        import sys
        sys.exit(1)
    ssgorb.append(cv2.ORB(nfeatures = 500))
    ssgorb[-1] = cv2.ORB_create(nfeatures = 500, edgeThreshold=5)
    ssgkp.append(ssgorb[i].detect(SSG[-1], None))
    def_, abc = ssgorb[i].compute(SSG[-1], ssgkp[-1])
    ssgdes.append(abc)

buckfm = cv2.BFMatcher(cv2.NORM_HAMMING)

orb = cv2.ORB(nfeatures = NUM_FEATURES, edgeThreshold=5)
orb = cv2.ORB_create(nfeatures = NUM_FEATURES, edgeThreshold=5)

def classify_image(image, image_id, classifier, reducer):
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    # kp to bitwise numpy array
    voting = [False] * 4

    for index, precompdes in enumerate(ssgdes):
        all_matches = buckfm.match(precompdes, des)
        all_matches.sort(key= lambda match: match.distance)

        if index == 0:
            dist_req = 30
        elif index == 1:
            dist_req = 40
        elif index == 2:
            dist_req = 40
        else:
            dist_req = 20
        matches = list(filter(lambda match: match.distance < dist_req, all_matches))
        
        if index == 0:
            match_req = 10
        if index == 1:
            match_req = 4
        elif index == 2:
            match_req = 3
        else:
            match_req = 3
        voting[index] = len(matches) >= match_req
        if index == 2:
            break

    outImg = np.zeros((1000,1000,3), np.uint8)
    outImg = cv2.drawMatches(SSG[index], ssgkp[index], image, kp, matches, outImg=outImg, flags=0)
    cv2.imshow('matching!', outImg)
    cv2.waitKey(300)

    vote_count = 0
    for b in voting:
        if b:
            vote_count += 1
    print(voting)
    if vote_count >= 2:
        print('stopsign!')
        return True
    else:
        print('meh')
        return False

def colorize_image(img, image_id, classifier, reducer):
    # else make edge green
    has_stopsign = classify_image(img, image_id, classifier, reducer)

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
    filename = OUT_BASE_STRING % (image_id,)
    cv2.imwrite(filename, img)

if __name__ == '__main__':
    # load data from csv, split into training and test sets
    classifier = joblib.load(KLASSIFIER_PATH)  
    reducer = joblib.load(REDUCER_PATH)
    print('loaded')
    for video_id in range(1, 2):
        print('video')
        for image_id in range(start_image_id, end_image_id, 2): # end_image_id):
            if image_id % 1 == 0:
                print('%02d %d / %d' % (1, image_id, end_image_id,))
            og_img = get_image(1, image_id)
            if og_img is None:
                print('og_img is None')
                continue
            else:
                og_img = og_img[:500]
            # print('colorize image')
            noise_img = colorize_image(og_img, image_id, classifier, reducer)
            # set_image(noise_img, 1, image_id)
