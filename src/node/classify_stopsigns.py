#!/usr/bin/env python
import rospy

import cv2
import joblib
import rospkg
import platform

from cv_bridge import CvBridge, CvBridgeError
from imblearn.under_sampling import RandomUnderSampler
from sensor_msgs.msg import Image
from sklearn.neighbors import KNeighborsClassifier
from std_msgs.msg import Bool

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

KLASSIFIER_PATH = '%s/data/017_the_500/competition_classifier01_%s.pkl' % (pkg_path, platform.python_version(),)
REDUCER_PATH = '%s/data/017_the_500/competition_reducer01_%s.pkl' % (pkg_path, platform.python_version(),)
classifier = joblib.load(KLASSIFIER_PATH)
reducer = joblib.load(REDUCER_PATH)

NUM_FEATURES = 500

orb = cv2.ORB_create(nfeatures = NUM_FEATURES)

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

def classify_image(image):
    kp = orb.detect(image, None)
    kp, des = orb.compute(img, kp)
    # kp to bitwise numpy array
    X = des
    smol_X = reducer.transform(X)
    y = classifier.predict(smol_X)

    # classify image based on match count
    if np.sum(y) > 10:
        # publish true on stopsign channel
        pub_buddy.publish(Bool(True))
    else:
        # publish false on stopsign channel
        pub_buddy.publish(Bool(False))


if __name__ == '__main__':
    image_in = rospy.Subscriber('/camera/image/rgb', Image, image_cb)
    rospy.init_node('find_me_stopsigns')
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if global_cv_image is not None:
            print('processing CV image')
            classify_image(image)
        else:
            print('no CV image recieved in last cycle')
        rate.sleep()
