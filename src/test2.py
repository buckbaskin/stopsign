import cv2
import datetime
import numpy as np

import math

from stopsign_core import StopsignFinder
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

finder = StopsignFinder()

# img_names = ['2mLeft', '2mRight', '3mLeft', '3mRight', '4mLeft', '4mRight']
img_names = ['2mLeft']

DEBUG = True

for img_name in img_names:
    img = cv2.imread('processed/'+img_name+'v1_keyp.jpg')

    if DEBUG:
        cv2.imshow('original image', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    result = finder.blob_detect(img, debug=DEBUG)

    filtered_result = []

    for item in result:
        if not (
            math.isnan(item.size) or
            math.isnan(item.pt[0]) or
            math.isnan(item.pt[1])):
            if item.size >= 12.0:
                filtered_result.append(item)

                # print('item: %s' % (item,))
                print('angle: %s' % (item.angle,))
                print('class_id: %s' % (item.class_id,))
                print('octave: %s' % (item.octave,))
                print('pt: %s' % (item.pt,))
                print('response: %s' % (item.response,))
                print('size: %s' % (item.size,))

    img_with_keypoints = cv2.drawKeypoints(
        img,
        result,
        np.array([]),
        (0,0,255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if DEBUG:
        cv2.imshow('Negative with keyp', img_with_keypoints)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # Results stopsign

    print('img: %s' % (img_name,))
    if len(result) > 0:
        print('Blobs found!')
    else:
        print('No blobs for you.')
