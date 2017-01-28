import cv2
import datetime

from stopsign_core import StopsignFinder
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

finder = StopsignFinder()

img_names = ['2mLeft.jpg', '2mRight.jpg', '3mLeft.jpg', '3mRight.jpg',
    '4mLeft.jpg', '4mRight.jpg', 'yellowv1.jpg', 'yellowv0.jpg']
img_names = ['v0.jpg', 'v1.jpg', 'v10.jpg', 'v11.jpg']

time_deltas = 0.0
img_count = len(img_names)

DEBUG = True
SAVE = False

TRIM = False

for img_name in img_names:
    start = datetime.datetime.now()

    img = cv2.imread('images/'+img_name)

    if TRIM:
        img_x = img.shape[0]
        img_y = img.shape[1]

        # Take away the top 3rd and bottom 3rd of the image
        img = img[360:(img_x-360)]

    if DEBUG:
        cv2.imshow('original image', img)
        cv2.waitKey()

    if SAVE:
        save_name = img_name[:-4]+'v1'
    else:
        save_name = False

    result = finder.check_for_stopsign(unwrap=False, img=img, debug=DEBUG, save=save_name)

    print('img: %s' % (img_name,))
    if result:
        print('Yeah! Stopsign!')
    else:
        print('Woops. This probably should be true')

    end = datetime.datetime.now()
    time_deltas += (end-start).total_seconds()

print('estimate frame rate: %s' % (float(img_count) / time_deltas,))
