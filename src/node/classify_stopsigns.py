#!/usr/bin/env python
import rospy

import cv2
import joblib
import rospkg
import platform
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

NUM_FEATURES = 500

orb = cv2.ORB(nfeatures = NUM_FEATURES)
preset = np.zeros((NUM_FEATURES, 256,))

bridge = CvBridge()

global_cv_image = None

def image_cb(image_msg):
    header = image_msg.header
    h = image_msg.height
    w = image_msg.width
    encoding = image_msg.encoding
    is_bigendian = image_msg.is_bigendian
    step = image_msg.step
    image_data = image_msg.data
    try:
        cv_image = bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding='passthrough',
            )
        global global_cv_image
        global_cv_image = cv_image
    except CvBridgeError as cvbe:
        print cvbe
        return None

pub_buddy = rospy.Publisher('/stopsign', Bool, queue_size=3)
GREY_STOPSIGN = '%s/data/019_stopsign_images/stopper%s.jpg' % (pkg_path, '%d')
NUM_FEATURES = 500

SSG = []
ssgorb = []
ssgkp = []
ssgdes = []
for i in range(4):
    SSG.append(cv2.imread(GREY_STOPSIGN % i, cv2.IMREAD_COLOR))
    if SSG[-1] is None:
        print('Image not loaded')
        import sys
        sys.exit(1)
    ssgorb.append(cv2.ORB(nfeatures = 500))
    # ssgorb[-1] = cv2.ORB_create(nfeatures = 500, edgeThreshold=5)
    ssgkp.append(ssgorb[i].detect(SSG[-1], None))
    def_, abc = ssgorb[i].compute(SSG[-1], ssgkp[-1])
    ssgdes.append(abc)

buckfm = cv2.BFMatcher(cv2.NORM_HAMMING)

orb = cv2.ORB(nfeatures = NUM_FEATURES, edgeThreshold=5)
# orb = cv2.ORB_create(nfeatures = NUM_FEATURES, edgeThreshold=5)


def classify_image(image):
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    # kp to bitwise numpy array
    voting = [False] * 4

    for index, precompdes in enumerate(ssgdes):
        all_matches = buckfm.match(precompdes, des)
        all_matches.sort(key= lambda match: match.distance)
	print(all_matches[0].distance)

        if index == 0:
            dist_req = 30
        elif index == 1:
            dist_req = 40
        elif index == 2:
            dist_req = 40
        else:
            dist_req = 20
        matches = list(filter(lambda match: match.distance < dist_req, all_matches))
	print('list %d' % (len(matches),))
        
        if index == 0:
            match_req = 10
        if index == 1:
            match_req = 3
        elif index == 2:
            match_req = 3
        else:
            match_req = 3
        voting[index] = len(matches) >= match_req

    vote_count = 0
    for b in voting:
        if b:
            vote_count += 1
    print(voting)
    if vote_count >= 1:
        # publish true on stopsign channel
        pub_buddy.publish(Bool(True))
        print('stopsign!')
        return True
    else:
        # publish false on stopsign channel
        pub_buddy.publish(Bool(False))
        print('meh')
        return False


if __name__ == '__main__':
    image_in = rospy.Subscriber('/camera/image', Image, image_cb)
    rospy.init_node('find_me_stopsigns')
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        if global_cv_image is not None:
            print('processing CV image')
            classify_image(global_cv_image)
	    global global_cv_image
	    global_cv_image = None
        else:
            print('no CV image recieved in last cycle')
        rate.sleep()
