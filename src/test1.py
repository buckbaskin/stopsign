import cv2

from stopsign_core import StopsignFinder
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

finder = StopsignFinder()

img_names = ['2mLeft.jpg', '2mRight.jpg', '3mLeft.jpg', '3mRight.jpg', '4mLeft.jpg', '4mRight.jpg']

for img_name in img_names:
    img = cv2.imread('images/'+img_name)
    result = finder.check_for_stopsign(unwrap=False, img=img)
    if result:
        print('Yeah! Stopsign!')
    else:
        print('Woops. This probably should be true')

