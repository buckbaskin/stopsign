#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

rospy.init_node('play_from_disc')
topic_out = '/camera/image'
imPub = rospy.Publisher(topic_out,Image,queue_size=1)
bridge = CvBridge()
base = 'frame%04d.jpg'
r = rospy.Rate(10)
for i in range(0, 900):
    if rospy.is_shutdown():
        break
    print(i)
    cvImage = cv2.imread(base % i, 1)
    if cvImage is not None:
        rosImageMsg = bridge.cv2_to_imgmsg(cvImage,encoding="bgr8")
        rosImageMsg.header.stamp = rospy.get_rostime()
        imPub.publish(rosImageMsg)
        r.sleep()
    else:
        print("Image not found. %s" % (base % i,))

print('done!')
