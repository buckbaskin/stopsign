#!/usr/bin/env python
import rospkg

import cv2
import numpy as np
from matplotlib import pyplot as plt

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

IMAGE_RATE = 11 # hz

EXACT_FILE = '%s/data/003_manual_labels/exact.csv' % (pkg_path,)
ALL_FILE = '%s/data/003_manual_labels/all.csv' % (pkg_path,)

start_image_id = 1550
end_image_id = 2189

IMAGE_BASE_STRING = '%s/data/002_original_images/%s' % (pkg_path, 'frame%04d.jpg')

def get_image(image_id):
    filename = IMAGE_BASE_STRING % (image_id,)
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def flatten_kp(kp):
    v = np.array(np.zeros((7,)))
    v[0] = kp.angle * 1000
    v[1] = kp.class_id
    v[2] = kp.octave
    v[3] = kp.pt[0]
    v[4] = kp.pt[1]
    v[5] = kp.response * 100000000
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

def kp_des2vector(klass, image_id, kp, des):
    vector = np.zeros((32+7+1+1,))
    vector[:1] = np.array([klass]) * 1000
    vector[-1] = np.array([image_id])
    vector[-8:-1] = np.array(flatten_kp(kp))
    vector[1:33] = des
    return vector

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
    val = cv2.waitKey(0) % 256
    test_kp = val == ord('s')
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
        vector = kp_des2vector(klass, image_id, kp[index], descriptor)
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
        vector = kp_des2vector(klass, image_id, kp[index], descriptor)
        results.append(vector)
    return results

def extend_file(file, new_vectors):
    for vector in new_vectors:
        file.write(','.join(['%7.2f' % num for num in vector]) + '\n')

def expand_to_string(new_vectors):
    for vec in new_vectors:
        yield ','.join(['%7d' % num for num in vec])

### Begin the whole process ###

# Generate the first line from data
line0 = []
line0.append('class'.ljust(7))
for i in range(32):
    line0.append('descr%02d' % (i,))
# line0.extend(['Keypoint Angle', 'Keypoint Class Id', 'Keypoint Octave', 'Keypoint X', 'Keypoint Y', 'Keypoint Response x 10^6', 'Keypoint Size'])
line0.extend(['angle'.ljust(7), 'classid', 'octave'.ljust(7), 'x'.ljust(7), 'y'.ljust(7), 'respons', 'size'.ljust(7)])
line0.append('imageid')
line0 = ','.join(line0)

exact_lines = [line0]
all_lines = [line0]

# Label all images before first stopsign as not-stopsign
print('Prefilling data')
for auto_image_id in range(start_image_id):
    if auto_image_id % 100 == 0:
        print('%d / %d' % (auto_image_id, start_image_id,))
    new_vectors = auto_label_image(auto_image_id, 0)
    all_lines.extend(expand_to_string(new_vectors))

print('Done Prefilling Data')

# Hand label sampled images and auto fill the rest
for image_id in range(start_image_id, end_image_id, IMAGE_RATE):
    new_vectors, is_stopsign = hand_label_image(image_id)
    exact_lines.extend(expand_to_string(new_vectors))
    all_lines.extend(expand_to_string(new_vectors))
    print('Autofilling data')
    for auto_image_id in range(image_id+1, image_id+IMAGE_RATE):
        new_vectors = auto_label_image(auto_image_id, 0.75 if is_stopsign else 0.25)
        all_lines.extend(expand_to_string(new_vectors))
    print('Done Autofilling data')


print('Write to EXACT_FILE')
with open(EXACT_FILE, 'w') as f:
    for line in exact_lines:
        f.write('%s\n' % (line,))
print('Write to ALL_FILE')
with open(ALL_FILE, 'w') as f:
    for line in all_lines:
        f.write('%s\n' % (line,))

