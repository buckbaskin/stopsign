import cv2
import datetime

from stopsign_core import StopsignFinder
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

finder = StopsignFinder()

img_names = ['2mLeft.jpg', '2mRight.jpg', '3mLeft.jpg', '3mRight.jpg', '4mLeft.jpg', '4mRight.jpg']
# img_names = ['2mLeft.jpg']

time_deltas = 0.0
img_count = len(img_names)

DEBUG = False

for img_name in img_names:
    start = datetime.datetime.now()

    img = cv2.imread('images/'+img_name)
    if DEBUG:
        cv2.imshow('original image', img)
        cv2.waitKey()
    result = finder.check_for_stopsign(unwrap=False, img=img, debug=DEBUG)
    print('img: %s' % (img_name,))
    if result:
        print('Yeah! Stopsign!')
    else:
        print('Woops. This probably should be true')

    end = datetime.datetime.now()
    time_deltas += (end-start).total_seconds()

print('estimate frame rate: %s' % (float(img_count) / time_deltas,))
