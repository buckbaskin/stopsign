import cv2
import datetime
import numpy as np

import math

from stopsign_core import StopsignFinder
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

finder = StopsignFinder()

img_names = ['2mLeft', '2mRight', '3mLeft', '3mRight', '4mLeft', '4mRight']
# img_names = ['2mLeft']

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
            filtered_result.append(item)

    # Results stopsign

    print('img: %s' % (img_name,))
    if len(result) > 0:
        print('Blobs found!')
    else:
        print('No blobs for you.')
