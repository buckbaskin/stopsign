#!/usr/bin/env python
import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from stopsign_core import StopsignFinder

FINDER = StopsignFinder()

OUT = rospy.Publisher('stopsign/detected', Bool, queue_size=1)
# PUB_MASK = rospy.Publisher('stopsign/mask', Image, queue_size=1)
PUB_MASK = None

try:
    CAP = cv2.VideoCapture(0)
except:
    CAP = None

BRIDGE = CvBridge()

def active_image():
    if CAP is not None:
        cv_image = CAP.read()[1]
        mess = Bool()
        mess.data = FINDER.check_for_stopsign(unwrap=False, img=cv_image, debug=False, save=False)
        OUT.publish(mess)

def image_cb(image_msg):
    # header = image_msg.header
    # h = image_msg.height
    # w = image_msg.width
    # encoding = image_msg.encoding
    # is_bigendian = image_msg.is_bigendian
    # uint32_step = image_msg.step
    # image_data = image_msg.data

    try:
        cv_image = BRIDGE.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        mess = Bool()
        mess.data = FINDER.check_for_stopsign(unwrap=False, img=cv_image, debug=False, save=False, pub_mask=PUB_MASK)
        try:
            OUT.publish(mess)
        except:
            pass
    except CvBridgeError as cvbe:
        print(cvbe)
    
    return None

IMAGE_IN = rospy.Subscriber('/camera/image', Image, image_cb)

if __name__ == '__main__':
    rospy.init_node('stopsign')
    print('rospy.init_node(stopsign)')
    if CAP is None:
        print('Failed to Capture Image.')
        rospy.loginfo('Waiting on messages')
        rospy.spin()
    else:
        rospy.loginfo('Waiting on messages')
        while not rospy.is_shutdown():
            active_image()
