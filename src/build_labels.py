#!/usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt

START_TIME = 70 # sec
IMAGE_RATE = 31 # hz
END_TIME = 120

OUT_FILE = '/home/buck/ros_ws/src/stopsign/data/generated_vectors.csv'

start_image_id = 1580
end_image_id = 2189

IMAGE_BASE_STRING = '/home/buck/ros_ws/src/stopsign/src/raw/frame%04d.jpg'

def get_image(image_id):
    filename = IMAGE_BASE_STRING % (image_id,)
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def flatten_kp(kp):
    v = np.array(np.zeros((7,)))
    v[0] = kp.angle
    v[1] = kp.class_id
    v[2] = kp.octave
    v[3] = kp.pt[0]
    v[4] = kp.pt[1]
    v[5] = kp.response * 1000000
    v[6] = kp.size
    return v

minx = 0
miny = 0
maxx = 10000
maxy = 10000
contour = []

def rebuild_contour():
    global minx, miny, maxx, maxy
    x1 = minx
    x2 = int(2.0/3 * minx + 1.0/3 * maxx)
    x3 = int(1.0/3 * minx + 2.0/3 * maxx)
    x4 = maxx
    y1 = miny
    y2 = int(2.0/3 * miny + 1.0/3 * maxy)
    y3 = int(1.0/3 * miny + 2.0/3 * maxy)
    y4 = maxy
    global contour
    contour = np.array([[x2, y1], [x3, y1], [x4, y2], [x4, y3],
                        [x3, y4], [x2, y4], [x1, y3], [x1, y2]], np.int32)

rebuild_contour()

def click_and_crop(event, x, y, flags, param):
    global minx, miny, maxx, maxy
    if event == cv2.EVENT_LBUTTONDOWN:
        minx = x
        miny = y
    elif event == cv2.EVENT_LBUTTONUP:
        maxx = x
        maxy = y
    rebuild_contour()


def hand_label_image(image_id):
    global minx, miny, maxx, maxy, contour
    results = []
    img = get_image(image_id)
    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    print('=====\npreview %04d\n' % (image_id,))
    print('s -> image has a stopsign.\nUse mouse to select stopsign.')
    print('\nOR\n')
    print('n -> image does not have a stopsign')
    print('---')
    cv2.imshow('preview', img)
    cv2.setMouseCallback('preview', click_and_crop)
    val = cv2.waitKey(0)
    test_kp = val == 1048691
    cv2.destroyAllWindows()

    if test_kp:
        for i in range(20):
            print('s -> accept polyline as region\n\nOR\n')
            print('Use mouse to reselect the region')
            print('n -> refresh polyline as region')
            print('---')
            imgur = cv2.drawKeypoints(
                img,
                filter(lambda x: cv2.pointPolygonTest(contour, x.pt, False) >= 0, kp),
                color=(0,255,0),
                flags=0)
            cv2.polylines(imgur, [contour], True, (79*i % 255, 0, 255))
            cv2.imshow('preview', imgur)
            cv2.setMouseCallback('preview', click_and_crop)
            val = cv2.waitKey(0)
            if val == 1048691:
                break

    cv2.destroyAllWindows()

    for index, keypoint in enumerate(kp):
        descriptor = des[index]
        if test_kp:
            skip_because_of_radius = cv2.pointPolygonTest(contour, kp[index].pt, False) < 0

            if not skip_because_of_radius:
                img2 = cv2.drawKeypoints(img, [kp[index]], color=(0,255,0), flags=4)
                if val == 1048691:
                    klass = 1
                elif val ==  1048686:
                    klass = 0
                else:
                    cv2.destroyAllWindows()
                    raise NotImplementedError('Use s or n please.')
            else:
                klass = 0
        else:
            klass = 0
        vector = np.zeros((32+7+1+1,))
        vector[-1:] = np.array([klass])
        vector[-2:-1] = np.array([image_id])
        vector[-9:-2] = np.array(flatten_kp(kp[index]))
        vector[0:32] = descriptor
        results.append(vector)
    cv2.destroyAllWindows()
    minx = 0
    miny = 0
    maxx = 10000
    maxy = 10000
    return results, test_kp

def auto_label_image(image_id, klass):
    results = []
    img = get_image(image_id)
    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    for index, keypoint in enumerate(kp):
        descriptor = des[index]
        vector = np.zeros((32+7+1+1,))
        vector[-1:] = np.array([klass])
        vector[-2:-1] = np.array([image_id])
        vector[-9:-2] = np.array(flatten_kp(kp[index]))
        vector[0:32] = descriptor
        results.append(vector)
    return results

def extend_file(file, new_vectors):
    for vector in new_vectors:
        file.write(','.join(['%7.2f' % num for num in vector]) + '\n')

with open(OUT_FILE, 'w') as f:
    line0 = []
    for i in range(32):
        line0.append('descr%02d' % (i,))
    line0.extend(['angle'.ljust(7), 'classid', 'octave'.ljust(7), 'x'.ljust(7), 'y'.ljust(7), 'respons', 'size'.ljust(7)])
    line0.append('imageid')
    line0.append('class'.ljust(7))
    f.write(','.join(line0) + '\n')
    print('Prefilling data')
    for auto_image_id in range(start_image_id):
        new_vectors = auto_label_image(auto_image_id, 0)
        extend_file(f, new_vectors)

    print('Done Prefilling Data')

    for image_id in range(start_image_id, end_image_id, 100):
        new_vectors, is_stopsign = hand_label_image(image_id)
        extend_file(f, new_vectors)
        print('Autofilling data')
        for auto_image_id in range(image_id+1, image_id+IMAGE_RATE):
            new_vectors = auto_label_image(auto_image_id, 1 if is_stopsign else 0)
            extend_file(f, new_vectors)
        print('Done Autofilling data')