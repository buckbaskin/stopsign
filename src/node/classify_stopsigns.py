#!/usr/bin/env python
import rospy

import cv2
import joblib
import rospkg

from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

MODEL_STORE_FILE = '%s/classifiers/mlp_classifier.pkl' % (pkg_path,)
classifier = joblib.load(MODEL_STORE_FILE)

ORB = cv2.createORB()

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


def classify_image(image):
    kp = ORB.fit(image)
    kp, des = ORB.describe(image, kp)

    # kp to bitwise numpy array
    X = kp
    y = classifier.predict(X)

    # filter down to matches

    # classify image based on match count
    if len(y) > 10:
        # publish true on stopsign channel
        pass
    else:
        # publish false on stopsign channel
        pass


if __name__ == '__main__':
    ### Begin the whole process ###

    '''
    Things to work on:
    Vary up the dataset:
        - Classify the total image instead of just one keypoint
            - Learn based on the classification of all of the keypoints in the 
            image and their location
    '''
    image_in = rospy.Subscriber('/camera/image/rgb', Image, image_cb)
    rospy.init_node('find_me_stopsigns')
    while rospy.isnt_dead():
        rospy.spinOnce()
        if global_cv_image is not None:
            classify_image(image)
